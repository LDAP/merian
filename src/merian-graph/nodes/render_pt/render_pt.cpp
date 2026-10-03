#include "merian-graph/nodes/render_pt/render_pt.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <type_traits>

namespace merian {

namespace {

std::string slot_module(const std::string& imports,
                        const std::string& guiding,
                        const std::string& volume_guiding,
                        const std::string& distance_guiding,
                        const std::string& irradiance_cache) {
    return fmt::format("module render_pt_guiding;\n"
                       "import merian_shaders.sampling.guiding;\n"
                       "import merian_shaders.light_cache.irradiance_cache;\n"
                       "{}"
                       "public typealias RenderGuiding = {};\n"
                       "public typealias RenderVolumeGuiding = {};\n"
                       "public typealias RenderDistanceGuiding = {};\n"
                       "public typealias RenderIrradianceCache = {};",
                       imports, guiding, volume_guiding, distance_guiding, irradiance_cache);
}

} // namespace

RenderPT::RenderPT()
    : guiding(std::make_shared<NullGuidingModel>()),
      distance_guiding(std::make_shared<NullDistanceGuidingModel>()),
      irradiance_cache(std::make_shared<NullIrradianceCache>()) {}

DeviceSupportInfo RenderPT::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_composition(GBufferLayout::complete()->get_composition());
    composition->add_module_from_string(
        "render_pt_guiding",
        slot_module("", "merian::NullGuidingModel", "merian::NullGuidingModel",
                    "merian::NullDistanceGuidingModel", "merian::NullIrradianceCache"));
    path_records.add_constants(composition);
    composition->add_module_from_path("merian-graph/nodes/render_pt/render_pt.slang", true);
    composition->add_module_from_path("merian-graph/nodes/render_pt/render_pt_volume.slang", true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline"},
                                    {"rayQuery", "rayTracingInvocationReorder"}) &
           program.get()->query_device_support(query_info);
}

void RenderPT::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    // the ray tracing pipeline caps the registers on AMD
    raygen_preferred = !context->get_device()->get_physical_device()->is_amd();
}

std::vector<InputConnectorDescriptor> RenderPT::describe_inputs() {
    std::vector<GBufferGroup> gbuffer_groups = {{{GBufferField::Hit}}};
    if (demodulate_albedo) {
        gbuffer_groups.push_back({{GBufferField::Albedo}});
    }
    if (volume_available) {
        gbuffer_groups.push_back({{GBufferField::LinearZ}});
        gbuffer_groups.push_back({{GBufferField::MotionVectors}});
    }
    con_gbuffer = GBufferIn::create(gbuffer_groups);

    return {
        {.name = "scene", .connector = con_scene},
        {.name = "gbuffer", .connector = con_gbuffer, .access = ConnectorAccess::ray_tracing_read},
        {.name = "guiding",
         .connector = con_guiding,
         .access = ConnectorAccess::ray_tracing_read,
         .optional = true},
        {.name = "distance_guiding",
         .connector = con_distance_guiding,
         .access = ConnectorAccess::compute_read,
         .optional = true},
        {.name = "irradiance_cache",
         .connector = con_irradiance_cache,
         .access = ConnectorAccess::ray_tracing_read,
         .optional = true},
        {.name = "prev_volume_depth",
         .connector = con_prev_volume_depth,
         .access = ConnectorAccess::compute_read,
         .delay = 1,
         .optional = true}};
}

std::vector<OutputConnectorDescriptor> RenderPT::describe_outputs(const NodeIOLayout& io_layout) {
    extent = io_layout[con_gbuffer]->get_create_info().extent;
    const auto adopt = [&](const auto& con, auto& slot, uint32_t& slot_version, const auto& none) {
        const bool is_connected = io_layout.is_connected(con);
        const std::remove_cvref_t<decltype(slot)> connected =
            is_connected ? io_layout[con]->get_create_info().model : none;
        const uint32_t version = is_connected ? io_layout[con]->get_create_info().version : 0;
        if (!slot || slot->get_type_name() != connected->get_type_name() ||
            slot_version != version) {
            composition.reset();
        }
        slot = connected;
        slot_version = version;
    };
    adopt(con_guiding, guiding, guiding_version, std::make_shared<NullGuidingModel>());
    adopt(con_distance_guiding, distance_guiding, distance_guiding_version,
          std::make_shared<NullDistanceGuidingModel>());
    adopt(con_irradiance_cache, irradiance_cache, irradiance_cache_version,
          std::make_shared<NullIrradianceCache>());
    con_irradiance = ManagedVkImageOut::create(irradiance_format, extent);
    con_debug = ManagedVkImageOut::create(vk::Format::eR32G32B32A32Sfloat, extent);
    con_volume = ManagedVkImageOut::create(irradiance_format, extent);
    con_volume_depth = ManagedVkImageOut::create(volume_depth_format, extent);
    con_volume_mv = ManagedVkImageOut::create(vk::Format::eR16G16Sfloat, extent);

    const bool no_volume = !volume_available;
    return {{.name = "irradiance",
             .connector = con_irradiance,
             .access = ConnectorAccess::ray_tracing_write},
            {.name = "debug",
             .connector = con_debug,
             .access = ConnectorAccess::ray_tracing_write | ConnectorAccess::compute_write},
            {.name = "volume",
             .connector = con_volume,
             .access = ConnectorAccess::compute_write,
             .disabled = no_volume},
            {.name = "volume_depth",
             .connector = con_volume_depth,
             .access = ConnectorAccess::compute_write,
             .disabled = no_volume},
            {.name = "volume_mv",
             .connector = con_volume_mv,
             .access = ConnectorAccess::compute_read_write,
             .disabled = no_volume},
            path_records.describe_output(context, extent, static_cast<uint32_t>(spp),
                                         recorded_vertices_per_path())};
}

RenderPT::NodeStatusFlags RenderPT::on_connected(const NodeIOLayout& io_layout,
                                                 const NodeIO& io,
                                                 [[maybe_unused]] const NodeConnectionInfo& info,
                                                 [[maybe_unused]] Submission& submission) {

    composition = nullptr;
    debug_connected = io_layout.is_connected(con_debug);

    if (path_records.update_connected(io_layout)) {
        return NEEDS_RECONNECT;
    }

    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            if (composition) {
                if (std::any_cast<bool>(force)) {
                    composition->force_reload();
                } else {
                    composition->reload(compile_context->get_search_path_file_loader());
                }
            }
            return true;
        });

    gbuffer_composition = io[con_gbuffer]->get_layout()->get_composition();

    if (const SceneHandle& scene = io[con_scene]; scene && scene->is_ready()) {
        ensure_pipeline(scene);
    }

    return {};
}

void RenderPT::update_guiding_slot() {
    std::string imports;
    const auto add = [&](const auto& model) {
        composition->add_composition(model->get_composition());
        for (const std::string& import : model->get_slang_imports()) {
            imports += fmt::format("import {};\n", import);
        }
    };
    add(guiding);
    add(distance_guiding);
    add(irradiance_cache);
    composition->add_module_from_string(
        "render_pt_guiding",
        slot_module(imports, guiding->get_type_name(),
                    guiding->supports_volume() ? guiding->get_type_name()
                                               : "merian::NullGuidingModel",
                    distance_guiding->get_type_name(), irradiance_cache->get_type_name()));
}

void RenderPT::ensure_pipeline(const SceneHandle& scene) {
    if (composition) {
        return;
    }
    composition = SlangComposition::create();
    composition->add_composition(scene->get_composition());
    composition->add_composition(gbuffer_composition);
    update_guiding_slot();
    composition->add_module_from_path("merian-graph/nodes/render_pt/render_pt.slang", true);
    composition->add_module_from_path("merian-graph/nodes/render_pt/render_pt_volume.slang", true);
    update_render_constants();

    program = SlangProgram::create(compile_context, composition);
    entry_point = SlangProgramEntryPoint::create(program, use_raygen() ? "main" : "main_compute");

    if (use_raygen()) {
        pipeline = Versioned<Pipeline>([this] {
            const auto ep = entry_point.get();
            return RayTracingPipelineBuilder()
                .add_raygen_group(ep->specialize())
                .build(ep->get_pipeline_layout(context));
        });
        pipeline.depends_on(entry_point);

        sbt = Versioned<ShaderBindingTable>([this] {
            return ShaderBindingTable::create(
                std::dynamic_pointer_cast<RayTracingPipeline>(pipeline.get()), resource_allocator);
        });
        sbt.depends_on(pipeline);
    } else {
        pipeline = Versioned<Pipeline>([this] {
            const auto ep = entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        pipeline.depends_on(entry_point);
    }

    params = Versioned<ShaderObject>([this] {
        return entry_point->create_shader_object_for_parameter(context, "params",
                                                               resource_allocator);
    });

    const auto build_volume_pass = [this](VolumePass& pass, const std::string& name) {
        pass.entry_point = SlangProgramEntryPoint::create(program, name);
        pass.pipeline = Versioned<Pipeline>([&pass, this] {
            const auto ep = pass.entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        pass.pipeline.depends_on(pass.entry_point);
        pass.params = Versioned<ShaderObject>([&pass, this] {
            return pass.entry_point->create_shader_object_for_parameter(context, "params",
                                                                        resource_allocator);
        });
        pass.params.depends_on(pass.entry_point);
    };
    build_volume_pass(single_scattering, "single_scattering");
    build_volume_pass(project_seed, "volume_project_seed");
    build_volume_pass(project, "volume_project");
    params.depends_on(entry_point);
}

[[nodiscard]] RenderPT::NodeStatusFlags
RenderPT::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const auto& cmd = submission.get_cmd();
    const auto& scene = io[con_scene];
    const auto gbuf = io[con_gbuffer];
    if (!scene || !scene->is_ready())
        return {};

    if (max_path_length != emitted_max_path_length) {
        emitted_max_path_length = max_path_length;
        io.send_event("bounces_changed");
    }

    if (const bool available = volume_spp > 0 && scene->has_exterior_volume();
        available != volume_available) {
        volume_available = available;
        return NodeStatusFlagBits::NEEDS_RECONNECT;
    }

    ensure_pipeline(scene);
    const ShaderObjectAllocatorHandle& obj_allocator = info.get_shader_object_allocator();

    const auto ep = entry_point.get();
    const auto pipe = pipeline.get();
    const auto params_obj = params.get();

    auto cursor = params_obj->get_cursor();
    cursor["gbuffer"] = gbuf.r();
    cursor["irradiance"] = io[con_irradiance].get_texture();
    cursor["debug"] = io[con_debug].get_texture();
    if (io.is_connected(con_guiding)) {
        cursor["guiding"] = io[con_guiding].r();
    }
    if (io.is_connected(con_irradiance_cache)) {
        cursor["irradiance_cache"] = io[con_irradiance_cache].r();
    }
    path_records.process(cmd, io, info.get_iteration(), cursor.find("path_records"));

    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
    ep->bind("params", params_obj, cmd, pipe, obj_allocator);

    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "surface");
        if (use_raygen()) {
            cmd->trace_rays(sbt.get(), extent);
        } else {
            cmd->dispatch(extent, 8, 8);
        }
    }

    const bool debugs_volume = debug_connected && debug_output == DebugOutput::DistanceGuiding;
    if (!volume_available ||
        !(debugs_volume || io.is_connected(con_volume) || io.is_connected(con_volume_depth) ||
          io.is_connected(con_volume_mv))) {
        return {};
    }

    const auto write_volume_binding = [&](const ShaderObjectHandle& obj) {
        auto volume_cursor = obj->get_cursor();
        volume_cursor["gbuffer"] = gbuf.r();
        volume_cursor["volume"] = io[con_volume].get_texture();
        volume_cursor["volume_depth"] = io[con_volume_depth].get_texture();
        volume_cursor["volume_mv"] = io[con_volume_mv].get_texture();
        volume_cursor["debug"] = io[con_debug].get_texture();
        if (io.is_connected(con_guiding) && guiding->supports_volume()) {
            volume_cursor["guiding"] = io[con_guiding].r();
        }
        if (io.is_connected(con_distance_guiding)) {
            volume_cursor["distance_guiding"] = io[con_distance_guiding].r();
        }
        if (io.is_connected(con_irradiance_cache)) {
            volume_cursor["irradiance_cache"] = io[con_irradiance_cache].r();
        }
        // resources do not convert to descriptors: assigning them falls into the byte-copy
        // overload, so bind the texture explicitly
        if (auto field = volume_cursor["prev_volume_depth"];
            field.is_valid() && io.is_connected(con_prev_volume_depth)) {
            field.write(io[con_prev_volume_depth].get_texture(), vk::ImageLayout::eGeneral);
        }
        return obj;
    };

    const auto dispatch_volume = [&](const VolumePass& pass, const bool bind_scene) {
        const auto volume_ep = pass.entry_point.get();
        const auto volume_pipe = pass.pipeline.get();
        cmd->bind(volume_pipe);
        if (bind_scene) {
            volume_ep->bind("scene", scene->get_shader_object(), cmd, volume_pipe, obj_allocator);
        }
        volume_ep->bind("params", write_volume_binding(pass.params.get()), cmd, volume_pipe,
                        obj_allocator);
        cmd->dispatch(extent, 8, 8);
    };

    const auto barrier_volume_mv = [&] {
        cmd->barrier(io[con_volume_mv]->barrier2(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::PipelineStageFlagBits2::eComputeShader));
    };

    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "volume project");
        dispatch_volume(project_seed, false);
        barrier_volume_mv();

        if (volume_forward_project && io.is_connected(con_prev_volume_depth) &&
            info.get_iteration() != 0) {
            dispatch_volume(project, true);
            barrier_volume_mv();
        }
    }

    if (debugs_volume) {
        cmd->barrier(io[con_debug]->barrier2(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
            vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eAllCommands,
            vk::PipelineStageFlagBits2::eComputeShader));
    }

    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "single scattering");
        dispatch_volume(single_scattering, true);
    }
    return {};
}

void RenderPT::update_render_constants() {
    uint32_t mask = 0u;
    for (uint32_t bit = 0; bit < 8; ++bit) {
        if (mask_enabled[bit])
            mask |= (1u << bit);
    }

    const DebugOutput emitted_debug_output = debug_connected ? debug_output : DebugOutput::None;
    const std::string constants = fmt::format(
        "import \"merian-graph/nodes/render_pt/render_pt_common.slang\";\n"
        "namespace merian {{\n"
        "export static const bool merian_render_emission_on_primary = {};\n"
        "export static const DebugOutput merian_render_debug_output = DebugOutput({});\n"
        "export static const bool merian_render_follow_specular = {};\n"
        "export static const float merian_render_follow_max_alpha = {:f};\n"
        "export static const int merian_render_spp = {};\n"
        "export static const uint merian_render_seed = {}u;\n"
        "export static const int merian_render_max_path_length = {};\n"
        "export static const uint merian_render_instance_mask = {}u;\n"
        "export static const bool merian_render_enable_ser = {};\n"
        "export static const bool merian_render_demodulate_albedo = {};\n"
        "export static const bool merian_render_russian_roulette = {};\n"
        "export static const ScatterMode merian_render_scatter_mode = ScatterMode({});\n"
        "export static const int merian_render_scatter_candidates = {};\n"
        "export static const bool merian_render_guiding_scale_with_alpha = {};\n"
        "export static const float merian_render_guiding_alpha_threshold = {:f};\n"
        "export static const GuidingDirectTarget merian_render_guiding_direct_target = "
        "GuidingDirectTarget({});\n"
        "export static const NEEMode merian_render_surface_nee_mode = NEEMode({});\n"
        "export static const float merian_render_surface_nee_probability = {:f};\n"
        "export static const int merian_render_surface_nee_bounces = {};\n"
        "export static const float merian_render_surface_guiding_share = {:f};\n"
        "export static const bool merian_render_surface_cache_tail = {};\n"
        "export static const int merian_render_volume_spp = {};\n"
        "export static const NEEMode merian_render_volume_nee_mode = NEEMode({});\n"
        "export static const float merian_render_volume_nee_probability = {:f};\n"
        "export static const float merian_render_volume_guiding_share = {:f};\n"
        "export static const bool merian_render_volume_cache_tail = {};\n"
        "export static const float merian_render_distance_guiding_share = {:f};\n"
        "export static const float merian_render_volume_forward_project_min_z = {:f};\n"
        "}}",
        emission_on_primary, static_cast<int32_t>(emitted_debug_output), follow_specular,
        follow_max_alpha, spp, seed, max_path_length, mask, enable_ser, demodulate_albedo,
        russian_roulette, static_cast<int32_t>(scatter_mode), scatter_candidates,
        guiding_scale_with_alpha, guiding_alpha_threshold,
        static_cast<int32_t>(guiding_direct_target), static_cast<int32_t>(surface.nee_mode),
        surface.nee_probability(has_guiding()), surface_nee_bounces,
        surface.guided_probability(has_guiding()), surface.cache_tail && has_irradiance_cache(),
        volume_spp, static_cast<int32_t>(volume.nee_mode),
        volume.nee_probability(has_volume_guiding()),
        volume.guided_probability(has_volume_guiding()),
        volume.cache_tail && has_irradiance_cache(), distance_guided_probability(),
        volume_forward_project_min_z);
    composition->add_module_from_string("render_pt_constants", constants);
    path_records.add_constants(composition);
}

float RenderPT::Sampling::nee_probability(const bool guided) const {
    if (nee_mode != NEEMode::Mixture) {
        return 0.f;
    }
    const float total = shading_share + nee_share + (guided ? guiding_share : 0.f);
    return total > 0.f ? nee_share / total : 0.f;
}

float RenderPT::Sampling::guided_probability(const bool guided) const {
    if (!guided) {
        return 0.f;
    }
    const float total = shading_share + guiding_share;
    return total > 0.f ? guiding_share / total : 0.f;
}

bool RenderPT::Sampling::properties(Properties& config,
                                    const std::string& shading_label,
                                    const bool guided,
                                    const bool has_cache) {
    bool changed = false;
    int nee_mode_index = static_cast<int>(nee_mode);
    if (config.config_options("nee", nee_mode_index, {"off", "mixture (MIS)", "resampled (RIS)"},
                              Properties::OptionsStyle::COMBO,
                              "Direct light sampling. 'mixture' replaces the scatter sample with a "
                              "light sample and costs no extra ray; 'resampled' adds a shadow "
                              "ray. How a light is chosen is the scene's to set.")) {
        nee_mode = static_cast<NEEMode>(nee_mode_index);
        changed = true;
    }
    changed |= config.config_split(
        "shares",
        {{shading_label, &shading_share},
         {"nee", &nee_share, nee_mode == NEEMode::Mixture},
         {"guiding", &guiding_share, guided}},
        "How the scatter samples are split between the shading function, the lights and the "
        "guiding method. The guiding method passes its share on to the shading function where it "
        "found nothing.");
    if (has_cache) {
        changed |= config.config_bool(
            "cache tail", cache_tail,
            "End a path that reaches the maximum length in the connected irradiance cache.");
    }
    return changed;
}

RenderPT::NodeStatusFlags RenderPT::properties(Properties& config) {
    bool needs_reconnect = false;
    bool constants_changed = false;
    // a configuration loads before the inputs connect, so only the UI hides what they decide
    const bool shows_guiding = has_guiding() || !config.is_ui();
    const bool shows_cache = has_irradiance_cache() || !config.is_ui();

    constants_changed |= config.config_uint(
        "seed", seed,
        "Decorrelates this run from another run of the same configuration. The sample stream "
        "is a function of the pixel, the frame and the sample index alone, so a reference and "
        "the images judged against it must not share it.");

    if (config.st_begin_child("surface", "Surface", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        constants_changed |=
            config.config_int("samples per pixel", spp, "Number of paths per pixel.", 1, 16);
        constants_changed |=
            config.config_int("max path length", max_path_length,
                              "Maximum number of path segments, including the primary hit.", 1, 16);
        constants_changed |=
            config.config_bool("russian roulette", russian_roulette,
                               "Terminate paths in proportion to the light they can still carry.");
        constants_changed |= surface.properties(config, "bsdf", has_guiding(), shows_cache);
        if (surface.nee_mode != NEEMode::Off) {
            constants_changed |= config.config_int(
                "nee bounces", surface_nee_bounces,
                "Path depth, counted from the primary hit and including the followed specular "
                "surfaces, up to which vertices sample lights; 0 = all.",
                0, 16);
        }
        config.st_end_child();
    }

    if (config.st_begin_child("volume", "Volume")) {
        constants_changed |= config.config_int(
            "samples per pixel", volume_spp,
            "Single-scattering samples along the primary ray; 0 disables the volume pass, and "
            "with it every node that consumes its outputs.",
            0, 16);
        if (volume_spp > 0) {
            constants_changed |=
                volume.properties(config, "phase", has_volume_guiding(), shows_cache);
            if (!std::dynamic_pointer_cast<NullDistanceGuidingModel>(distance_guiding) ||
                !config.is_ui()) {
                constants_changed |= config.config_split(
                    "distance shares",
                    {{"transmittance", &distance_transmittance_share},
                     {"guiding", &distance_guiding_share}},
                    "How the scattering distances are split between the transmittance and the "
                    "distance guiding method.");
            }
            config.config_bool(
                "forward project", volume_forward_project,
                "Reproject the mean scattering distance into this frame's motion vectors instead "
                "of keeping the surface ones, which describe the first opaque hit.");
            if (volume_forward_project) {
                constants_changed |= config.config_float(
                    "forward project min z", volume_forward_project_min_z,
                    "Below this scattering distance the surface motion vector is the better "
                    "estimate.",
                    1.f, 0.f);
            }
        }
        config.st_end_child();
    }

    if (config.st_begin_child("guiding", "Guiding")) {
        int scatter_mode_index = static_cast<int>(scatter_mode);
        if (config.config_options(
                "sampling", scatter_mode_index, {"off", "mixture (MIS)", "resampled (RIS)"},
                Properties::OptionsStyle::COMBO,
                "How one direction comes out of the guiding lobes and the shading function. "
                "'off' draws from the shading function alone; 'resampled' draws several and keeps "
                "one by how much the shading function makes of it, at the cost of the extra "
                "evaluations; it still traces one ray.")) {
            scatter_mode = static_cast<ScatterMode>(scatter_mode_index);
            constants_changed = true;
        }
        if (scatter_mode == ScatterMode::RIS) {
            constants_changed |= config.config_int("candidates", scatter_candidates,
                                                   "Directions drawn before one is kept.", 1, 16);
        }
        constants_changed |= config.config_bool(
            "follow specular", follow_specular,
            "Follow a smooth surface instead of making it a path vertex: no guiding lobe and no "
            "light sampler resolves it, so it costs a guiding query and a write for nothing, and "
            "leaves the vertex before it aiming at the surface rather than at the light behind "
            "it.");
        if (follow_specular) {
            constants_changed |= config.config_float(
                "follow below alpha", follow_max_alpha,
                "Surfaces with a smaller GGX alpha (roughness squared) are followed.", 0.001f, 0.f,
                1.f);
        }
        if (shows_guiding) {
            constants_changed |= config.config_bool(
                "scale with alpha", guiding_scale_with_alpha,
                "Scale the guiding share with the GGX alpha of the surface, so a narrow lobe "
                "keeps its own sampling.");
            constants_changed |= config.config_float(
                "alpha threshold", guiding_alpha_threshold,
                "Below this GGX alpha the guiding lobes are broader than the shading function "
                "itself, so nothing is guided.",
                0.001f, 0.f, 1.f);
            int direct_target_index = static_cast<int>(guiding_direct_target);
            if (config.config_options(
                    "direct light target", direct_target_index, {"full", "MIS", "none"},
                    Properties::OptionsStyle::COMBO,
                    "What a method learns from a vertex that ended on a light: the emission "
                    "whole, only the share the scatter technique pays for, or nothing.")) {
                guiding_direct_target = static_cast<DirectTarget>(direct_target_index);
                constants_changed = true;
            }
            const float p_nee = surface.nee_probability(true);
            const float p_guided = (1.f - p_nee) * surface.guided_probability(true) *
                                   (guiding_scale_with_alpha ? 0.5f : 1.f);
            config.output_text(fmt::format(
                "at alpha 0.5: {:.0f} % guided, {:.0f} % lights, {:.0f} % shading function",
                p_guided * 100.f, p_nee * 100.f, (1.f - p_nee - p_guided) * 100.f));
        }
        config.st_end_child();
    }

    if (config.st_begin_child("instance_mask", "Instance mask")) {
        for (uint32_t bit = 0; bit < 8; ++bit) {
            constants_changed |= config.config_bool(std::to_string(bit), mask_enabled[bit]);
            if ((bit & 3u) != 3u)
                config.st_no_space();
        }
        config.st_end_child();
    }

    if (config.st_begin_child("output", "Output")) {
        constants_changed |=
            config.config_bool("emission on primary", emission_on_primary,
                               "Fold primary-hit emission into irradiance (self-contained). "
                               "Otherwise it is the GBuffer emission texture's job.");
        needs_reconnect |= config.config_bool(
            "demodulate albedo", demodulate_albedo,
            "Divide the primary-hit albedo out of the output so a denoiser can re-modulate after "
            "filtering. Use with 'emission on primary' disabled (emission is albedo-independent).");
        int debug_output_index = static_cast<int>(debug_output);
        if (config.config_options(
                "debug output", debug_output_index,
                {"scatter statistics", "guiding debug", "nee debug", "distance guiding debug"},
                Properties::OptionsStyle::COMBO,
                "What the debug output holds, where a node consumes it. The scatter statistics "
                "count per pixel the scatter samples the guiding drew, the ones the shading "
                "function refused and the ones the geometry refused a ray; the guiding methods "
                "and the scene pick their own debug views.")) {
            debug_output = static_cast<DebugOutput>(debug_output_index);
            constants_changed = true;
        }
        needs_reconnect |= config.config_enum("irradiance format", irradiance_format,
                                              Properties::OptionsStyle::COMBO);
        config.st_end_child();
    }

    if (config.st_begin_child("other", "Other")) {
        if (use_raygen()) {
            constants_changed |=
                config.config_bool("shader execution reordering", enable_ser,
                                   "Reorder threads after the primary hit to improve coherence.");
        }
        int trace_shader_index = static_cast<int>(trace_shader);
        if (config.config_options(
                "trace shader", trace_shader_index, {"auto", "ray generation", "compute"},
                Properties::OptionsStyle::COMBO,
                "Trace from a ray generation shader or a compute shader. The compute shader avoids "
                "the ray tracing pipeline's register cap; 'auto' picks it on AMD.")) {
            trace_shader = static_cast<TraceShader>(trace_shader_index);
            needs_reconnect = true;
        }
        config.st_end_child();
    }

    needs_reconnect |=
        path_records.properties(config, static_cast<uint32_t>(spp), recorded_vertices_per_path());

    if (constants_changed && composition) {
        update_render_constants();
    }

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

float RenderPT::distance_guided_probability() const {
    const float total = distance_transmittance_share + distance_guiding_share;
    return total > 0.f ? distance_guiding_share / total : 0.f;
}

bool RenderPT::has_guiding() const {
    return !std::dynamic_pointer_cast<NullGuidingModel>(guiding) &&
           scatter_mode != ScatterMode::Off;
}

bool RenderPT::has_volume_guiding() const {
    return has_guiding() && guiding->supports_volume();
}

bool RenderPT::has_irradiance_cache() const {
    return !std::dynamic_pointer_cast<NullIrradianceCache>(irradiance_cache);
}

bool RenderPT::use_raygen() const {
    return trace_shader == TraceShader::Auto ? raygen_preferred
                                             : trace_shader == TraceShader::RayGeneration;
}

uint32_t RenderPT::recorded_vertices_per_path() const {
    const auto length = static_cast<uint32_t>(max_path_length);
    return (surface.nee_mode == NEEMode::Resampled ? 2 * length : length) + 1;
}

} // namespace merian
