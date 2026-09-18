#include "merian-graph/nodes/render_pt/render_pt.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

namespace merian {

RenderPT::RenderPT()
    : guiding(std::make_shared<NullGuidingModel>()),
      distance_guiding(std::make_shared<NullDistanceGuidingModel>()) {}

DeviceSupportInfo RenderPT::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_composition(GBufferLayout::complete()->get_composition());
    composition->add_module_from_string(
        "render_pt_guiding",
        "module render_pt_guiding;\n"
        "import \"merian-shaders/sampling/guiding.slang\";\n"
        "public typealias RenderGuiding = merian::NullGuidingModel;\n"
        "public typealias RenderDistanceGuiding = merian::NullDistanceGuidingModel;");
    composition->add_module_from_path("merian-graph/nodes/render_pt/render_pt.slang", true);
    composition->add_module_from_path("merian-graph/nodes/render_pt/render_pt_volume.slang", true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline"}, {"rayQuery"}) &
           program.get()->query_device_support(query_info);
}

void RenderPT::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    use_raygen = !context->get_device()->get_physical_device()->is_amd();
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
        {.name = "prev_volume_depth",
         .connector = con_prev_volume_depth,
         .access = ConnectorAccess::compute_read,
         .delay = 1,
         .optional = true}};
}

std::vector<OutputConnectorDescriptor> RenderPT::describe_outputs(const NodeIOLayout& io_layout) {
    extent = io_layout[con_gbuffer]->get_create_info().extent;
    // the method's type and constants are baked in, so either changing rebuilds the program
    const auto adopt = [&](const ShaderObjectInHandle<GuidingObject>& con, GuidingModelHandle& slot,
                           uint32_t& slot_version, const GuidingModelHandle& none) {
        const GuidingModelHandle connected =
            io_layout.is_connected(con) ? io_layout[con]->get_create_info().model : none;
        const uint32_t version =
            io_layout.is_connected(con) ? io_layout[con]->get_create_info().version : 0;
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
    con_irradiance = ManagedVkImageOut::create(irradiance_format, extent);
    con_volume = ManagedVkImageOut::create(irradiance_format, extent);
    con_volume_depth = ManagedVkImageOut::create(volume_depth_format, extent);
    con_volume_mv = ManagedVkImageOut::create(vk::Format::eR16G16Sfloat, extent);

    const bool no_volume = !volume_available;
    return {{.name = "irradiance",
             .connector = con_irradiance,
             .access = ConnectorAccess::ray_tracing_write},
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
             .disabled = no_volume}};
}

RenderPT::NodeStatusFlags RenderPT::on_connected(const NodeIOLayout& io_layout,
                                                 const NodeIO& io,
                                                 [[maybe_unused]] const NodeConnectionInfo& info,
                                                 [[maybe_unused]] Submission& submission) {

    // force the program graph to be rewired next process()
    composition = nullptr;

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
    for (const GuidingModelHandle& model : {guiding, distance_guiding}) {
        composition->add_composition(model->get_composition());
        for (const std::string& import : model->get_slang_imports()) {
            imports += fmt::format("import {};\n", import);
        }
    }
    composition->add_module_from_string(
        "render_pt_guiding",
        fmt::format("module render_pt_guiding;\n"
                    "import \"merian-shaders/sampling/guiding.slang\";\n"
                    "{}"
                    "public typealias RenderGuiding = {};\n"
                    "public typealias RenderDistanceGuiding = {};",
                    imports, guiding->get_type_name(), distance_guiding->get_type_name()));
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
    entry_point = SlangProgramEntryPoint::create(program, use_raygen ? "main" : "main_compute");

    if (use_raygen) {
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
    if (io.is_connected(con_guiding)) {
        cursor["guiding"] = io[con_guiding].r();
    }

    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
    ep->bind("params", params_obj, cmd, pipe, obj_allocator);

    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "surface");
        if (use_raygen) {
            cmd->trace_rays(sbt.get(), extent);
        } else {
            cmd->dispatch(extent, 8, 8);
        }
    }

    if (!volume_available) {
        return {};
    }

    const auto write_volume_binding = [&](const ShaderObjectHandle& obj) {
        auto volume_cursor = obj->get_cursor();
        volume_cursor["gbuffer"] = gbuf.r();
        volume_cursor["volume"] = io[con_volume].get_texture();
        volume_cursor["volume_depth"] = io[con_volume_depth].get_texture();
        volume_cursor["volume_mv"] = io[con_volume_mv].get_texture();
        if (io.is_connected(con_guiding)) {
            volume_cursor["guiding"] = io[con_guiding].r();
        }
        if (io.is_connected(con_distance_guiding)) {
            volume_cursor["distance_guiding"] = io[con_distance_guiding].r();
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

        // the ping-pong is only valid from the second iteration on
        if (volume_forward_project && io.is_connected(con_prev_volume_depth) &&
            info.get_iteration() != 0) {
            dispatch_volume(project, true);
            barrier_volume_mv();
        }
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

    const std::string constants =
        fmt::format("namespace merian {{\n"
                    "export static const bool merian_render_emission_on_primary = {};\n"
                    "export static const int merian_render_guiding_debug_view = {};\n"
                    "export static const bool merian_render_scatter_stats = {};\n"
                    "export static const bool merian_render_follow_specular = {};\n"
                    "export static const float merian_render_specular_alpha = {:f};\n"
                    "export static const int merian_render_spp = {};\n"
                    "export static const uint merian_render_seed = {}u;\n"
                    "export static const int merian_render_max_path_length = {};\n"
                    "export static const uint merian_render_instance_mask = {}u;\n"
                    "export static const bool merian_render_enable_ser = {};\n"
                    "export static const bool merian_render_demodulate_albedo = {};\n"
                    "export static const int merian_render_nee_mode = {};\n"
                    "export static const float merian_render_nee_probability = {:f};\n"
                    "export static const int merian_render_nee_bounces = {};\n"
                    "export static const int merian_render_scatter_mode = {};\n"
                    "export static const int merian_render_scatter_candidates = {};\n"
                    "export static const int merian_render_volume_spp = {};\n"
                    "export static const float merian_render_volume_forward_project_min_z "
                    "= {:f};\n"
                    "export static const float merian_guiding_share = {:f};\n"
                    "export static const bool merian_guiding_scale_with_alpha = {};\n"
                    "export static const float merian_guiding_alpha_threshold = {:f};\n"
                    "export static const int merian_guiding_direct_target = {};\n"
                    "export static const float merian_guiding_distance_share = {:f};\n"
                    "}}",
                    emission_on_primary ? "true" : "false", guiding_debug_view,
                    scatter_stats ? "true" : "false", follow_specular ? "true" : "false",
                    specular_alpha, spp, seed, max_path_length, mask, enable_ser ? "true" : "false",
                    demodulate_albedo ? "true" : "false", nee_mode, nee_probability, nee_bounces,
                    scatter_mode, scatter_candidates, volume_spp, volume_forward_project_min_z,
                    guiding_share, guiding_scale_with_alpha ? "true" : "false",
                    guiding_alpha_threshold, guiding_direct_target, guiding_distance_share);
    SPDLOG_INFO("render_pt constants:\n{}", constants);
    composition->add_module_from_string("render_pt_constants", constants);
}

RenderPT::NodeStatusFlags RenderPT::properties(Properties& config) {
    bool needs_reconnect = false;
    bool constants_changed = false;

    constants_changed |=
        config.config_int("samples per pixel", spp, "Number of paths per pixel.", 1, 16);
    constants_changed |= config.config_uint(
        "seed", seed,
        "Decorrelates this run from another run of the same configuration. The sample stream "
        "is a function of the pixel, the frame and the sample index alone, so a reference and "
        "the images judged against it must not share it.");
    constants_changed |=
        config.config_int("max path length", max_path_length,
                          "Maximum number of path segments, including the primary hit.", 1, 16);
    constants_changed |=
        config.config_bool("emission on primary", emission_on_primary,
                           "Fold primary-hit emission into irradiance (self-contained). "
                           "Otherwise it is the GBuffer emission texture's job.");

    if (config.st_begin_child("scatter", "Scatter")) {
        constants_changed |= config.config_options(
            "sampling", scatter_mode, {"mixture (MIS)", "resampled (RIS)"},
            Properties::OptionsStyle::COMBO,
            "How one direction comes out of the guiding lobes and the shading function. "
            "'resampled' draws several and keeps one by how much the shading function makes of "
            "it, at the cost of the extra evaluations; it still traces one ray.");
        if (scatter_mode == 1) {
            constants_changed |= config.config_int("candidates", scatter_candidates,
                                                   "Directions drawn before one is kept.", 1, 16);
        }
        constants_changed |= config.config_bool(
            "follow specular", follow_specular,
            "Follow a surface that scatters singularly instead of making it a path vertex: it "
            "cannot be guided and no light sampler can reach it, so it costs a guiding query and "
            "a write for nothing, and leaves the vertex before it aiming at the surface rather "
            "than at the light behind it.");
        if (follow_specular) {
            constants_changed |= config.config_float(
                "follow below alpha", specular_alpha,
                "Also follow surfaces narrower than this, which do not scatter singularly. 0 "
                "follows only the singular ones.",
                0.01f, 0.f, 1.f);
        }
        config.st_end_child();
    }

    if (config.st_begin_child("nee", "Next event estimation")) {
        constants_changed |= config.config_options(
            "mode", nee_mode, {"off", "mixture", "resampled"}, Properties::OptionsStyle::COMBO,
            "Direct light sampling. 'mixture' replaces the scatter sample "
            "with a light sample and costs no extra ray; 'resampled' adds a "
            "shadow ray. How a light is chosen is the scene's to set.");
        if (nee_mode == 1) {
            constants_changed |= config.config_percent(
                "direct light share", nee_probability,
                "Fraction of scatter samples drawn from the lights. A light sample replaces the "
                "scatter sample, so this is taken out of the budget the indirect signal lives on. "
                "How far it pays depends on how good the light samples are.");
        }
        if (nee_mode != 0) {
            constants_changed |= config.config_int(
                "bounces", nee_bounces,
                "Path vertices (counted from the primary hit) that sample lights; 0 = all.", 0, 16);
            if (nee_mode == 1) {
                config.output_text(fmt::format("one draw: {:.0f} % light, {:.0f} % scatter",
                                               nee_probability * 100.f,
                                               (1.f - nee_probability) * 100.f));
            } else {
                config.output_text("candidates into one shadow ray: the scene's to split");
            }
        }
        config.st_end_child();
    }

    if (config.st_begin_child("guiding", "Guiding")) {
        if (!guiding) {
            config.output_text("connect a guiding method to the guiding slot");
        } else {
            constants_changed |= config.config_percent(
                "share", guiding_share,
                "Fraction of the scatter samples the guiding method gets, where it found "
                "something. The rest go to the shading function.");
            constants_changed |= config.config_bool(
                "scale with roughness", guiding_scale_with_alpha,
                "Scale that share with the lobe width, so a narrow lobe keeps its own sampling.");
            constants_changed |= config.config_float(
                "roughness threshold", guiding_alpha_threshold,
                "Below this lobe width the guiding lobes are broader than the shading function "
                "itself, so nothing is guided.",
                0.01f, 0.f, 1.f);
            constants_changed |= config.config_options(
                "direct light target", guiding_direct_target, {"full", "MIS", "none"},
                Properties::OptionsStyle::COMBO,
                "What a method learns from a vertex that ended on a light: the emission whole, "
                "only the share the scatter technique pays for, or nothing.");
            const float effective = guiding_scale_with_alpha ? guiding_share * 0.5f : guiding_share;
            config.output_text(
                fmt::format("at a lobe width of 0.5: {:.0f} % guided, {:.0f} % shading function",
                            effective * 100.f, (1.f - effective) * 100.f));
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
            if (distance_guiding) {
                constants_changed |= config.config_percent(
                    "distance guiding share", guiding_distance_share,
                    "Fraction of the distance samples the distance guiding method gets.");
            }
            needs_reconnect |= config.config_bool(
                "forward project", volume_forward_project,
                "Reproject the mean scattering distance into this frame's motion vectors instead "
                "of keeping the surface ones, which describe the first opaque hit.");
            if (volume_forward_project) {
                constants_changed |= config.config_float(
                    "forward project min z", volume_forward_project_min_z,
                    "Below this scattering distance the surface motion vector is the better "
                    "estimate.",
                    0.f);
            }
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
        needs_reconnect |= config.config_bool(
            "demodulate albedo", demodulate_albedo,
            "Divide the primary-hit albedo out of the output so a denoiser can re-modulate after "
            "filtering. Use with 'emission on primary' disabled (emission is albedo-independent).");
        constants_changed |= config.config_int(
            "guiding debug view", guiding_debug_view,
            "Render the guiding method's own view of its state instead of the image. 0 is off; "
            "what each index shows is listed by the guiding node.",
            0, 16);
        constants_changed |= config.config_bool(
            "scatter statistics", scatter_stats,
            "Replace the image with per-pixel counts of the scatter samples the guiding drew and "
            "the ones nothing could be traced from.");
        needs_reconnect |= config.config_enum("irradiance format", irradiance_format,
                                              Properties::OptionsStyle::COMBO);
        constants_changed |=
            config.config_bool("shader execution reordering", enable_ser,
                               "Reorder threads after the primary hit to improve coherence.");
        needs_reconnect |= config.config_bool(
            "ray tracing pipeline", use_raygen,
            "Trace from a raygen shader instead of a compute shader. The compute path avoids the "
            "ray-tracing pipeline register cap.");
        config.st_end_child();
    }

    if (constants_changed && composition) {
        update_render_constants();
    }

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
