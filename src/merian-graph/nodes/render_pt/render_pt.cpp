#include "merian-graph/nodes/render_pt/render_pt.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"

#include <fmt/format.h>

namespace merian {

RenderPT::RenderPT() = default;

DeviceSupportInfo RenderPT::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_module_from_path("merian-graph/nodes/render_pt/render_pt.slang", true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline"}, {"rayQuery"}) &
           program.get()->query_device_support(query_info);
}

void RenderPT::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();
}

std::vector<InputConnectorDescriptor> RenderPT::describe_inputs() {
    return {{"scene", con_scene}, {"gbuffer", con_gbuffer, ConnectorAccess::ray_tracing_read}};
}

std::vector<OutputConnectorDescriptor>
RenderPT::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_irradiance = ManagedVkImageOut::create(vk::Format::eR32G32B32A32Sfloat, extent);
    return {{"irradiance", con_irradiance, ConnectorAccess::ray_tracing_write}};
}

RenderPT::NodeStatusFlags RenderPT::on_connected(const NodeConnectedInfo& info) {
    const NodeIOLayout& io_layout = info.io_layout;

    // force the program graph to be rewired next process()
    composition = nullptr;
    obj_allocator = nullptr;

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

    return {};
}

void RenderPT::process(GraphRun& run, const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& scene = io[con_scene];
    const auto gbuf = io[con_gbuffer];
    if (!scene || !scene->is_ready())
        return;

    if (max_path_length != emitted_max_path_length) {
        emitted_max_path_length = max_path_length;
        io.send_event("bounces_changed");
    }

    if (!composition) {
        composition = SlangComposition::create();
        composition->add_composition(scene->get_composition());
        composition->add_module_from_path("merian-graph/nodes/render_pt/render_pt.slang", true);
        update_render_constants();

        program = SlangProgram::create(compile_context, composition);
        entry_point = SlangProgramEntryPoint::create(program, "main");

        pipeline = Versioned<RayTracingPipeline>([this] {
            const auto ep = entry_point.get();
            return RayTracingPipelineBuilder()
                .add_raygen_group(ep->specialize())
                .build(ep->get_pipeline_layout(context));
        });
        pipeline.depends_on(entry_point);

        sbt = Versioned<ShaderBindingTable>(
            [this] { return ShaderBindingTable::create(pipeline.get(), resource_allocator); });
        sbt.depends_on(pipeline);

        params = Versioned<ShaderObject>([this] {
            return entry_point->create_shader_object_for_parameter(context, "params",
                                                                   resource_allocator);
        });
        params.depends_on(entry_point);

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    obj_allocator->set_iteration(run.get_in_flight_index());

    const auto ep = entry_point.get();
    const auto pipe = pipeline.get();
    const auto params_obj = params.get();

    auto cursor = params_obj->get_cursor();
    cursor["gbuffer"] = gbuf.r();
    cursor["irradiance"] = io[con_irradiance].get_texture();

    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
    ep->bind("params", params_obj, cmd, pipe, obj_allocator);

    cmd->trace_rays(sbt.get(), extent);
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
                    "}}",
                    emission_on_primary ? "true" : "false", spp, max_path_length, mask,
                    enable_ser ? "true" : "false", demodulate_albedo ? "true" : "false"));
}

RenderPT::NodeStatusFlags RenderPT::properties(Properties& config) {
    bool needs_reconnect = false;
    bool constants_changed = false;

    constants_changed |= config.config_int("samples per pixel", spp,
                                           "Number of BSDF-sampled paths per pixel.", 1, 16);
    constants_changed |=
        config.config_int("max path length", max_path_length,
                          "Maximum number of path segments, including the primary hit.", 1, 16);
    constants_changed |=
        config.config_bool("emission on primary", emission_on_primary,
                           "Fold primary-hit emission into irradiance (self-contained). "
                           "Otherwise it is the GBuffer emission texture's job.");
    constants_changed |=
        config.config_bool("shader execution reordering", enable_ser,
                           "Reorder threads after the primary hit to improve coherence.");
    constants_changed |= config.config_bool(
        "demodulate albedo", demodulate_albedo,
        "Divide the primary-hit albedo out of the output so a denoiser can re-modulate after "
        "filtering. Use with 'emission on primary' disabled (emission is albedo-independent).");

    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

    config.st_separate("instance mask");
    for (uint32_t bit = 0; bit < 8; ++bit) {
        constants_changed |= config.config_bool(std::to_string(bit), mask_enabled[bit]);
        if ((bit & 3u) != 3u)
            config.st_no_space();
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
