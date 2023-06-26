#include "compute_node.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

ComputeNode::ComputeNode(const SharedContext context,
                         const ResourceAllocatorHandle allocator,
                         const std::optional<uint32_t> push_constant_size)
    : context(context), allocator(allocator), push_constant_size(push_constant_size) {}

void ComputeNode::cmd_build(const vk::CommandBuffer&,
                            const std::vector<std::vector<merian::ImageHandle>>& image_inputs,
                            const std::vector<std::vector<merian::BufferHandle>>& buffer_inputs,
                            const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                            const std::vector<std::vector<merian::BufferHandle>>& buffer_outputs) {

    std::tie(textures, sets, pool, layout) = make_graph_descriptor_sets(
        context, allocator, image_inputs, buffer_inputs, image_outputs, buffer_outputs, layout);

    if (!pipe) {
        auto pipe_builder = PipelineLayoutBuilder(context);
        if (push_constant_size.has_value()) {
            pipe_builder.add_push_constant(push_constant_size.value());
        }
        auto pipe_layout = pipe_builder.add_descriptor_set_layout(layout).build_pipeline_layout();
        pipe = std::make_shared<ComputePipeline>(pipe_layout, get_shader_module(),
                                                 get_specialization_info());
    }
}

void ComputeNode::cmd_process(const vk::CommandBuffer& cmd,
                              GraphRun&,
                              const uint32_t set_index,
                              const std::vector<ImageHandle>&,
                              const std::vector<BufferHandle>&,
                              const std::vector<ImageHandle>&,
                              const std::vector<BufferHandle>&) {
    pipe->bind(cmd);
    pipe->bind_descriptor_set(cmd, sets[set_index]);
    if (push_constant_size.has_value())
        pipe->push_constant(cmd, get_push_constant());
    auto [x, y, z] = get_group_count();
    cmd.dispatch(x, y, z);
}

} // namespace merian
