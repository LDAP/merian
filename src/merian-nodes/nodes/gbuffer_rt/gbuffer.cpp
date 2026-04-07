#include "merian-nodes/nodes/gbuffer_rt/gbuffer.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

GBufferRTNode::GBufferRTNode() {}

DeviceSupportInfo
GBufferRTNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return DeviceSupportInfo::check(
        query_info, {"accelerationStructure", "computeDerivativeGroupQuads"});
}

void GBufferRTNode::initialize(const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = ShaderCompileContext::create(context);
}

std::vector<InputConnectorDescriptor> GBufferRTNode::describe_inputs() {
    return {{"scene", con_scene}};
}

std::vector<OutputConnectorDescriptor>
GBufferRTNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_position =
        ManagedVkImageOut::compute_read_write(vk::Format::eR32G32B32A32Sfloat, extent);
    con_normal =
        ManagedVkImageOut::compute_read_write(vk::Format::eR32G32B32A32Sfloat, extent);
    con_albedo =
        ManagedVkImageOut::compute_read_write(vk::Format::eR8G8B8A8Unorm, extent);

    return {{"position", con_position}, {"normal", con_normal}, {"albedo", con_albedo}};
}

GBufferRTNode::NodeStatusFlags GBufferRTNode::on_connected(
    [[maybe_unused]] const NodeIOLayout& io_layout,
    [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {

    // Pipeline is created lazily in process() once the scene is available,
    // since we need the scene's Slang composition for link-time constants.
    pipeline = nullptr;
    program = nullptr;
    entry_point = nullptr;
    gbuffer_obj = nullptr;
    obj_allocator = nullptr;

    return {};
}

void GBufferRTNode::process(GraphRun& run,
                            [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                            const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& scene = io[con_scene];
    if (!scene || !scene->has_geometry())
        return;

    // Lazily build program/pipeline from scene's composition
    if (!pipeline) {
        auto composition = SlangComposition::create();
        composition->add_composition(scene->get_composition());
        composition->add_module_from_path("merian-nodes/nodes/gbuffer_rt/gbuffer.slang",
                                          true /* with entry points */);

        program = SlangProgram::create(compile_context, composition);
        entry_point = SlangProgramEntryPoint::create(program, "main");

        auto pipe_layout = entry_point->get_pipeline_layout(context);
        auto vulkan_entry_point = entry_point->specialize();
        pipeline = ComputePipeline::create(pipe_layout, vulkan_entry_point);

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    // Create/update GBuffer output object
    if (!gbuffer_obj) {
        gbuffer_obj = entry_point->create_shader_object(context, "gbuffer", obj_allocator);
    }

    auto cursor = gbuffer_obj->get_cursor();
    cursor["position"] = io[con_position].get_texture()->get_view();
    cursor["normal"] = io[con_normal].get_texture()->get_view();
    cursor["albedo"] = io[con_albedo].get_texture()->get_view();

    // Bind and dispatch
    obj_allocator->set_iteration(run.get_in_flight_index());
    cmd->bind(pipeline);
    entry_point->bind_entry_point_parameter("scene", scene->get_shader_object(), cmd, pipeline);
    entry_point->bind_entry_point_parameter("gbuffer", gbuffer_obj, cmd, pipeline);

    cmd->dispatch(extent, 16, 16);
}

GBufferRTNode::NodeStatusFlags GBufferRTNode::properties(Properties& config) {
    bool needs_reconnect = false;
    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
