#include "merian-graph/nodes/render_pt_mcpg/render_pt_mcpg.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"

#include <fmt/format.h>

namespace merian {

namespace {
struct PushConstant {
    int32_t spp;
    int32_t max_path_length;
    uint32_t instance_mask;
};
} // namespace

RenderMCPG::RenderMCPG() = default;

DeviceSupportInfo RenderMCPG::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/render_pt_mcpg.slang", true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline"}, {"rayQuery"}) &
           program.get()->query_device_support(query_info);
}

void RenderMCPG::initialize(const ContextHandle& context,
                            const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();
}

std::vector<InputConnectorDescriptor> RenderMCPG::describe_inputs() {
    return {{"scene", con_scene}, {"gbuffer", con_gbuffer}};
}

std::vector<OutputConnectorDescriptor>
RenderMCPG::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_irradiance = ManagedVkImageOut::compute_write(vk::Format::eR32G32B32A32Sfloat, extent);
    return {{"irradiance", con_irradiance}};
}

RenderMCPG::NodeStatusFlags
RenderMCPG::on_connected(const NodeIOLayout& io_layout,
                         [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {

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

void RenderMCPG::process(GraphRun& run,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& scene = io[con_scene];
    const auto& gbuf = io[con_gbuffer];
    if (!scene || !gbuf || !scene->is_ready())
        return;

    if (max_path_length != emitted_max_path_length) {
        emitted_max_path_length = max_path_length;
        io.send_event("bounces_changed");
    }

    if (!composition) {
        composition = SlangComposition::create();
        composition->add_composition(scene->get_composition());
        composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/render_pt_mcpg.slang",
                                          true);
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
    cursor["gbuffer"] = gbuf->get_shader_object();
    cursor["irradiance"] = io[con_irradiance].get_texture();

    uint32_t mask = 0u;
    for (uint32_t bit = 0; bit < 8; ++bit) {
        if (mask_enabled[bit])
            mask |= (1u << bit);
    }

    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
    ep->bind("params", params_obj, cmd, pipe, obj_allocator);
    cmd->push_constant(pipe, PushConstant{spp, max_path_length, mask});

    cmd->trace_rays(sbt.get(), extent);
}

void RenderMCPG::update_render_constants() {
    composition->add_module_from_string("render_pt_constants",
                                        fmt::format("namespace merian {{ export static const bool "
                                                    "merian_render_emission_on_primary = {}; }}",
                                                    emission_on_primary ? "true" : "false"));
}

RenderMCPG::NodeStatusFlags RenderMCPG::properties(Properties& config) {
    bool needs_reconnect = false;

    config.config_int("samples per pixel", spp, "Number of BSDF-sampled paths per pixel.", 1, 16);
    config.config_int("max path length", max_path_length,
                      "Maximum number of path segments, including the primary hit.", 1, 16);
    if (config.config_bool("emission on primary", emission_on_primary,
                           "Fold primary-hit emission into irradiance (self-contained). "
                           "Otherwise it is the GBuffer emission texture's job.") &&
        composition) {
        update_render_constants();
    }

    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

    config.st_separate("instance mask");
    for (uint32_t bit = 0; bit < 8; ++bit) {
        config.config_bool(std::to_string(bit), mask_enabled[bit]);
        if ((bit & 3u) != 3u)
            config.st_no_space();
    }

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
