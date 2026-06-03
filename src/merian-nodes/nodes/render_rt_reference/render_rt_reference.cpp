#include "merian-nodes/nodes/render_rt_reference/render_rt_reference.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

namespace {
struct PushConstant {
    int32_t spp;
    int32_t max_path_length;
};
} // namespace

RenderRTReferenceNode::RenderRTReferenceNode() = default;

DeviceSupportInfo
RenderRTReferenceNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return DeviceSupportInfo::check(query_info, {}, {"rayQuery"});
}

void RenderRTReferenceNode::initialize(const ContextHandle& context,
                                       const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = ShaderCompileContext::create(context);
}

std::vector<InputConnectorDescriptor> RenderRTReferenceNode::describe_inputs() {
    return {{"scene", con_scene}, {"gbuffer", con_gbuffer}};
}

std::vector<OutputConnectorDescriptor>
RenderRTReferenceNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_irradiance = ManagedVkImageOut::compute_write(vk::Format::eR32G32B32A32Sfloat, extent);
    return {{"irradiance", con_irradiance}};
}

RenderRTReferenceNode::NodeStatusFlags RenderRTReferenceNode::on_connected(
    [[maybe_unused]] const NodeIOLayout& io_layout,
    [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {

    pipeline = nullptr;
    program = nullptr;
    entry_point = nullptr;
    params = nullptr;
    obj_allocator = nullptr;

    return {};
}

void RenderRTReferenceNode::process(GraphRun& run,
                                    [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                                    const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& scene = io[con_scene];
    const auto& gbuf = io[con_gbuffer];
    if (!scene || !gbuf || !scene->is_ready())
        return;

    if (!program) {
        auto composition = SlangComposition::create();
        composition->add_composition(scene->get_composition());
        composition->add_module_from_path(
            "merian-nodes/nodes/render_rt_reference/render_rt_reference.slang", true);

        program = SlangProgram::create(compile_context, composition);
        entry_point = SlangProgramEntryPoint::create(program, "main");

        // entry_point rebuilds when the scene's composition (material type
        // conformances) changes — drop cached pipeline + per-EP shader objects.
        entry_point->on_changed(entry_point, [this] {
            pipeline = nullptr;
            params = nullptr;
        });

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    if (!pipeline) {
        auto pipe_layout = entry_point->get_pipeline_layout(context);
        auto vulkan_entry_point = entry_point->specialize();
        pipeline = ComputePipeline::create(pipe_layout, vulkan_entry_point);
    }

    obj_allocator->set_iteration(run.get_in_flight_index());

    if (!params) {
        params = entry_point->create_shader_object(context, "params", resource_allocator);
    }

    auto cursor = params->get_cursor();
    cursor["gbuffer"] = gbuf->get_shader_object();
    cursor["irradiance"] = io[con_irradiance].get_texture();

    cmd->bind(pipeline);
    entry_point->bind_entry_point_parameter("scene", scene->get_shader_object(), cmd, pipeline,
                                            obj_allocator);
    entry_point->bind_entry_point_parameter("params", params, cmd, pipeline, obj_allocator);
    cmd->push_constant(pipeline, PushConstant{spp, max_path_length});

    cmd->dispatch(extent, 16, 16);
}

RenderRTReferenceNode::NodeStatusFlags RenderRTReferenceNode::properties(Properties& config) {
    bool needs_reconnect = false;

    config.config_int("samples per pixel", spp, "Number of BSDF-sampled paths per pixel.", 1, 16);
    config.config_int("max path length", max_path_length,
                      "Maximum number of path segments, including the primary hit.", 1, 16);

    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
