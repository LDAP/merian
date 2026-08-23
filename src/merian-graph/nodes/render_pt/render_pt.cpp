#include "merian-graph/nodes/render_pt/render_pt.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"

#include <fmt/format.h>

namespace merian {

RenderPT::RenderPT() : guiding(std::make_shared<NullGuidingModel>()) {}

DeviceSupportInfo RenderPT::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_module_from_string(
        "render_pt_guiding", "module render_pt_guiding;\n"
                             "import \"merian-shaders/sampling/guiding.slang\";\n"
                             "public typealias RenderGuiding = merian::NullGuidingModel;");
    composition->add_module_from_path("merian-graph/nodes/render_pt/render_pt.slang", true);
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
    return {{"scene", con_scene},
            {"gbuffer", con_gbuffer, ConnectorAccess::ray_tracing_read},
            {.name = "guiding",
             .connector = con_guiding,
             .access = ConnectorAccess::ray_tracing_read,
             .optional = true}};
}

std::vector<OutputConnectorDescriptor> RenderPT::describe_outputs(const NodeIOLayout& io_layout) {
    extent = io_layout[con_gbuffer]->get_create_info().extent;
    const GuidingModelHandle connected =
        io_layout.is_connected(con_guiding)
            ? io_layout[con_guiding]->get_create_info().model
            : std::static_pointer_cast<GuidingModel>(std::make_shared<NullGuidingModel>());
    const uint32_t connected_version =
        io_layout.is_connected(con_guiding) ? io_layout[con_guiding]->get_create_info().version : 0;
    // the method's type and constants are baked in, so either changing rebuilds the program
    if (!guiding || guiding->get_type_name() != connected->get_type_name() ||
        guiding_version != connected_version) {
        composition.reset();
    }
    guiding = connected;
    guiding_version = connected_version;
    con_irradiance = ManagedVkImageOut::create(irradiance_format, extent);
    return {{"irradiance", con_irradiance, ConnectorAccess::ray_tracing_write}};
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

    if (const SceneHandle& scene = io[con_scene]; scene && scene->is_ready()) {
        ensure_pipeline(scene);
    }

    return {};
}

void RenderPT::update_guiding_slot() {
    std::string imports;
    for (const std::string& import : guiding->get_slang_imports()) {
        imports += fmt::format("import \"{}\";\n", import);
    }
    composition->add_composition(guiding->get_composition());
    composition->add_module_from_string(
        "render_pt_guiding", fmt::format("module render_pt_guiding;\n"
                                         "import \"merian-shaders/sampling/guiding.slang\";\n"
                                         "{}"
                                         "public typealias RenderGuiding = {};",
                                         imports, guiding->get_type_name()));
}

void RenderPT::ensure_pipeline(const SceneHandle& scene) {
    if (composition) {
        return;
    }
    composition = SlangComposition::create();
    composition->add_composition(scene->get_composition());
    update_guiding_slot();
    composition->add_module_from_path("merian-graph/nodes/render_pt/render_pt.slang", true);
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

    if (use_raygen) {
        cmd->trace_rays(sbt.get(), extent);
    } else {
        cmd->dispatch(extent, 8, 8);
    }
    return {};
}

void RenderPT::update_render_constants() {
    uint32_t mask = 0u;
    for (uint32_t bit = 0; bit < 8; ++bit) {
        if (mask_enabled[bit])
            mask |= (1u << bit);
    }

    composition->add_module_from_string(
        "render_pt_constants",
        fmt::format("namespace merian {{\n"
                    "export static const bool merian_render_emission_on_primary = {};\n"
                    "export static const int merian_render_spp = {};\n"
                    "export static const int merian_render_max_path_length = {};\n"
                    "export static const uint merian_render_instance_mask = {}u;\n"
                    "export static const bool merian_render_enable_ser = {};\n"
                    "export static const bool merian_render_demodulate_albedo = {};\n"
                    "export static const int merian_render_nee_mode = {};\n"
                    "export static const float merian_render_nee_probability = {:f};\n"
                    "export static const int merian_render_nee_candidates = {};\n"
                    "export static const int merian_render_nee_bounces = {};\n"
                    "}}",
                    emission_on_primary ? "true" : "false", spp, max_path_length, mask,
                    enable_ser ? "true" : "false", demodulate_albedo ? "true" : "false", nee_mode,
                    nee_probability, nee_candidates, nee_bounces));
}

RenderPT::NodeStatusFlags RenderPT::properties(Properties& config) {
    bool needs_reconnect = false;
    bool constants_changed = false;

    constants_changed |=
        config.config_int("samples per pixel", spp, "Number of paths per pixel.", 1, 16);
    constants_changed |=
        config.config_int("max path length", max_path_length,
                          "Maximum number of path segments, including the primary hit.", 1, 16);
    constants_changed |=
        config.config_bool("emission on primary", emission_on_primary,
                           "Fold primary-hit emission into irradiance (self-contained). "
                           "Otherwise it is the GBuffer emission texture's job.");
    constants_changed |=
        config.config_options("next event estimation", nee_mode, {"off", "mixture", "resampled"},
                              Properties::OptionsStyle::COMBO,
                              "Direct light sampling. 'mixture' replaces the "
                              "scatter sample with a light sample and costs no "
                              "extra ray; 'resampled' adds a shadow ray.");
    if (nee_mode == 1) {
        constants_changed |=
            config.config_percent("NEE probability", nee_probability,
                                  "Fraction of scatter samples drawn from the lights.");
    }
    if (nee_mode == 2) {
        constants_changed |= config.config_int(
            "NEE candidates", nee_candidates,
            "Light samples resampled into the one shadow ray by unshadowed contribution (RIS).", 1,
            32);
    }
    if (nee_mode != 0) {
        constants_changed |= config.config_int(
            "NEE bounces", nee_bounces,
            "Path vertices (counted from the primary hit) that perform NEE; 0 = all.", 0, 16);
    }
    constants_changed |=
        config.config_bool("shader execution reordering", enable_ser,
                           "Reorder threads after the primary hit to improve coherence.");
    constants_changed |= config.config_bool(
        "demodulate albedo", demodulate_albedo,
        "Divide the primary-hit albedo out of the output so a denoiser can re-modulate after "
        "filtering. Use with 'emission on primary' disabled (emission is albedo-independent).");

    config.st_separate("instance mask");
    for (uint32_t bit = 0; bit < 8; ++bit) {
        constants_changed |= config.config_bool(std::to_string(bit), mask_enabled[bit]);
        if ((bit & 3u) != 3u)
            config.st_no_space();
    }

    if (constants_changed && composition) {
        update_render_constants();
    }

    config.st_separate();
    needs_reconnect |= config.config_bool(
        "ray tracing pipeline", use_raygen,
        "Trace from a raygen shader instead of a compute shader. The compute path avoids the "
        "ray-tracing pipeline register cap.");
    needs_reconnect |=
        config.config_enum("irradiance format", irradiance_format, Properties::OptionsStyle::COMBO);

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
