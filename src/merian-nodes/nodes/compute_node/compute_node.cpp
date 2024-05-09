#include "compute_node.hpp"
#include "merian-nodes/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"

namespace merian {

ComputeNode::ComputeNode(const SharedContext context,
                         const ResourceAllocatorHandle allocator,
                         const std::optional<uint32_t> push_constant_size)
    : context(context), allocator(allocator), push_constant_size(push_constant_size) {}

void ComputeNode::cmd_build(const vk::CommandBuffer&, const std::vector<NodeIO>& ios) {

    std::tie(textures, sets, pool, layout) =
        make_graph_descriptor_sets(context, allocator, ios, layout);

    if (!pipe_layout) {
        auto pipe_builder = PipelineLayoutBuilder(context);
        if (push_constant_size.has_value()) {
            pipe_builder.add_push_constant(push_constant_size.value());
        }
        pipe_layout = pipe_builder.add_descriptor_set_layout(layout).build_pipeline_layout();
    }

    pipe = std::make_shared<ComputePipeline>(pipe_layout, get_shader_module(),
                                             get_specialization_info());
}

void ComputeNode::cmd_process(const vk::CommandBuffer& cmd,
                              GraphRun& run,
                              [[maybe_unused]] const std::shared_ptr<FrameData>& frame_data,
                              const uint32_t set_index,
                              [[maybe_unused]] const NodeIO& io) {
    pipe->bind(cmd);
    pipe->bind_descriptor_set(cmd, sets[set_index]);
    if (push_constant_size.has_value())
        pipe->push_constant(cmd, get_push_constant(run));
    auto [x, y, z] = get_group_count();
    cmd.dispatch(x, y, z);
}

} // namespace merian
