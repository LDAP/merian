#include "compute_node.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"

namespace merian_nodes {

ComputeNode::ComputeNode(const SharedContext context,
                         const std::string& name,
                         const std::optional<uint32_t> push_constant_size)
    : Node(name), context(context), push_constant_size(push_constant_size) {}

ComputeNode::NodeStatusFlags
ComputeNode::on_connected(const DescriptorSetLayoutHandle& descriptor_set_layout) {

    auto pipe_builder = PipelineLayoutBuilder(context);
    if (push_constant_size.has_value()) {
        pipe_builder.add_push_constant(push_constant_size.value());
    }
    pipe_layout =
        pipe_builder.add_descriptor_set_layout(descriptor_set_layout).build_pipeline_layout();

    pipe = std::make_shared<ComputePipeline>(pipe_layout, get_shader_module(),
                                             get_specialization_info());

    return {};
}

void ComputeNode::process(GraphRun& run,
                          const vk::CommandBuffer& cmd,
                          const DescriptorSetHandle& descriptor_set,
                          [[maybe_unused]] const ConnectorResourceMap& resource_for_connector,
                          [[maybe_unused]] std::any& in_flight_data) {
    pipe->bind(cmd);
    pipe->bind_descriptor_set(cmd, descriptor_set);
    if (push_constant_size.has_value())
        pipe->push_constant(cmd, get_push_constant(run));
    auto [x, y, z] = get_group_count();
    cmd.dispatch(x, y, z);
}

} // namespace merian_nodes
