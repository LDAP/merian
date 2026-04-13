#include "merian-nodes/nodes/gbuffer_debug/gbuffer_debug.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"

namespace merian {

GBufferDebugNode::GBufferDebugNode() {}

void GBufferDebugNode::initialize(const ContextHandle& context,
                                  const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = ShaderCompileContext::create(context);
}

std::vector<InputConnectorDescriptor> GBufferDebugNode::describe_inputs() {
    return {{"gbuffer", con_gbuffer}};
}

std::vector<OutputConnectorDescriptor>
GBufferDebugNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_output = ManagedVkImageOut::compute_write(vk::Format::eR8G8B8A8Unorm, extent);
    return {{"image", con_output}};
}

GBufferDebugNode::NodeStatusFlags GBufferDebugNode::on_connected(
    [[maybe_unused]] const NodeIOLayout& io_layout,
    [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {

    pipeline = nullptr;
    program = nullptr;
    entry_point = nullptr;
    debug_gbuffer_obj = nullptr;
    obj_allocator = nullptr;

    return {};
}

void GBufferDebugNode::process(GraphRun& run,
                               [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                               const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& gbuf = io[con_gbuffer];
    if (!gbuf)
        return;

    if (!pipeline) {
        auto composition = SlangComposition::create();
        composition->add_composition(gbuf->get_composition());
        composition->add_module_from_path("merian-nodes/nodes/gbuffer_debug/gbuffer_debug.slang",
                                          true);

        program = SlangProgram::create(compile_context, composition);
        entry_point = SlangProgramEntryPoint::create(program, "main");

        auto pipe_layout = entry_point->get_pipeline_layout(context);
        auto vulkan_entry_point = entry_point->specialize();
        pipeline = ComputePipeline::create(pipe_layout, vulkan_entry_point);

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    if (!debug_gbuffer_obj) {
        debug_gbuffer_obj =
            entry_point->create_shader_object(context, "gbuffer", obj_allocator);
    }

    auto cursor = debug_gbuffer_obj->get_cursor();
    cursor["tex0"] = gbuf->get_tex0()->get_view();
    cursor["tex1"] = gbuf->get_tex1()->get_view();
    cursor["tex2"] = gbuf->get_tex2()->get_view();
    cursor["output"] = io[con_output].get_texture()->get_view();

    obj_allocator->set_iteration(run.get_in_flight_index());
    cmd->bind(pipeline);
    entry_point->bind_entry_point_parameter("gbuffer", debug_gbuffer_obj, cmd, pipeline);
    cmd->push_constant(pipeline, static_cast<int>(selected_field));

    cmd->dispatch(extent, 16, 16);
}

GBufferDebugNode::NodeStatusFlags GBufferDebugNode::properties(Properties& config) {
    bool needs_reconnect = false;

    const std::vector<std::string> field_names = {
        "Normal",         "Linear Z",    "Grad Z",      "Delta Z",
        "Motion Vectors", "Instance ID", "Primitive ID", "Barycentrics",
    };
    int field_idx = static_cast<int>(selected_field);
    config.config_options("field", field_idx, field_names);
    selected_field = static_cast<FieldSelect>(field_idx);

    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
