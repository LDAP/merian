#include "median.hpp"
#include "merian-nodes/graph/graph.hpp"
#include "merian-nodes/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "median_histogram.comp.spv.h"
#include "median_reduce.comp.spv.h"

namespace merian {

MedianApproxNode::MedianApproxNode(const SharedContext context,
                                   const ResourceAllocatorHandle allocator,
                                   const int component)
    : context(context), allocator(allocator), component(component) {
    assert(component < 4 && component >= 0);

    histogram = std::make_shared<ShaderModule>(context, merian_median_histogram_comp_spv_size(),
                                               merian_median_histogram_comp_spv());
    reduce = std::make_shared<ShaderModule>(context, merian_median_reduce_comp_spv_size(),
                                            merian_median_reduce_comp_spv());
}

MedianApproxNode::~MedianApproxNode() {}

std::string MedianApproxNode::name() {
    return "Median Approximation";
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
MedianApproxNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("src"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
MedianApproxNode::describe_outputs(const std::vector<NodeOutputDescriptorImage>&,
                                   const std::vector<NodeOutputDescriptorBuffer>&) {
    return {
        {},
        {
            NodeOutputDescriptorBuffer(
                "median", vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::BufferCreateInfo({}, sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer)),
            NodeOutputDescriptorBuffer(
                "histogram", vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::BufferCreateInfo({}, local_size_x * local_size_y * sizeof(uint32_t),
                                     vk::BufferUsageFlagBits::eStorageBuffer)),
        },
    };
}

void MedianApproxNode::cmd_build([[maybe_unused]] const vk::CommandBuffer& cmd,
                                 const std::vector<NodeIO>& ios) {
    std::tie(graph_textures, graph_sets, graph_pool, graph_layout) =
        make_graph_descriptor_sets(context, allocator, ios, graph_layout);

    if (!pipe_reduce) {
        auto pipe_layout = PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(graph_layout)
                               .add_push_constant<PushConstant>()
                               .build_pipeline_layout();
        auto spec_builder = SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y, component);
        SpecializationInfoHandle spec = spec_builder.build();

        pipe_histogram = std::make_shared<ComputePipeline>(pipe_layout, histogram, spec);
        pipe_reduce = std::make_shared<ComputePipeline>(pipe_layout, reduce, spec);
    }
}

void MedianApproxNode::cmd_process(const vk::CommandBuffer& cmd,
                                   [[maybe_unused]] GraphRun& run,
                                   [[maybe_unused]] const std::shared_ptr<FrameData>& frame_data,
                                   const uint32_t set_index,
                                   const NodeIO& io) {
    const auto group_count_x =
        (io.image_inputs[0]->get_extent().width + local_size_x - 1) / local_size_x;
    const auto group_count_y =
        (io.image_inputs[0]->get_extent().height + local_size_y - 1) / local_size_y;

    cmd.fillBuffer(*io.buffer_outputs[1], 0, VK_WHOLE_SIZE, 0);
    auto bar = io.buffer_outputs[1]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                    vk::AccessFlagBits::eShaderRead |
                                                        vk::AccessFlagBits::eShaderWrite);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});

    pipe_histogram->bind(cmd);
    pipe_histogram->bind_descriptor_set(cmd, graph_sets[set_index]);
    pipe_histogram->push_constant(cmd, pc);
    cmd.dispatch(group_count_x, group_count_y, 1);

    bar = io.buffer_outputs[1]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                   vk::AccessFlagBits::eShaderWrite,
                                               vk::AccessFlagBits::eShaderRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});

    pipe_reduce->bind(cmd);
    pipe_reduce->bind_descriptor_set(cmd, graph_sets[set_index]);
    pipe_reduce->push_constant(cmd, pc);
    cmd.dispatch(1, 1, 1);
}

void MedianApproxNode::get_configuration(Configuration& config, bool&) {
    config.config_float("min", pc.min);
    config.config_float("max", pc.max);
}

} // namespace merian
