#include "median.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "median_histogram.comp.spv.h"
#include "median_reduce.comp.spv.h"

namespace merian_nodes {

MedianApproxNode::MedianApproxNode(const ContextHandle context) : Node(), context(context) {

    histogram = std::make_shared<ShaderModule>(context, merian_median_histogram_comp_spv_size(),
                                               merian_median_histogram_comp_spv());
    reduce = std::make_shared<ShaderModule>(context, merian_median_reduce_comp_spv_size(),
                                            merian_median_reduce_comp_spv());
}

MedianApproxNode::~MedianApproxNode() {}

std::vector<InputConnectorHandle> MedianApproxNode::describe_inputs() {
    return {con_src};
}

std::vector<OutputConnectorHandle>
MedianApproxNode::describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {

    con_median = std::make_shared<ManagedVkBufferOut>(
        "median", vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo({}, sizeof(float), vk::BufferUsageFlagBits::eStorageBuffer));
    con_histogram = std::make_shared<ManagedVkBufferOut>(
        "histogram", vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo({}, local_size_x * local_size_y * sizeof(uint32_t),
                             vk::BufferUsageFlagBits::eStorageBuffer));
    return {con_median, con_histogram};
}

MedianApproxNode::NodeStatusFlags
MedianApproxNode::on_connected(const DescriptorSetLayoutHandle& descriptor_set_layout) {
    if (!pipe_reduce) {
        auto pipe_layout = PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(descriptor_set_layout)
                               .add_push_constant<PushConstant>()
                               .build_pipeline_layout();
        auto spec_builder = SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y, component);
        SpecializationInfoHandle spec = spec_builder.build();

        pipe_histogram = std::make_shared<ComputePipeline>(pipe_layout, histogram, spec);
        pipe_reduce = std::make_shared<ComputePipeline>(pipe_layout, reduce, spec);
    }

    return {};
}

void MedianApproxNode::process([[maybe_unused]] GraphRun& run,
                               [[maybe_unused]] const vk::CommandBuffer& cmd,
                               [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                               [[maybe_unused]] const NodeIO& io) {
    const auto group_count_x = (io[con_src]->get_extent().width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (io[con_src]->get_extent().height + local_size_y - 1) / local_size_y;

    cmd.fillBuffer(*io[con_histogram], 0, VK_WHOLE_SIZE, 0);
    auto bar = io[con_histogram]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                 vk::AccessFlagBits::eShaderRead |
                                                     vk::AccessFlagBits::eShaderWrite);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});

    pipe_histogram->bind(cmd);
    pipe_histogram->bind_descriptor_set(cmd, descriptor_set);
    pipe_histogram->push_constant(cmd, pc);
    cmd.dispatch(group_count_x, group_count_y, 1);

    bar = io[con_histogram]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                vk::AccessFlagBits::eShaderWrite,
                                            vk::AccessFlagBits::eShaderRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});

    pipe_reduce->bind(cmd);
    pipe_reduce->bind_descriptor_set(cmd, descriptor_set);
    pipe_reduce->push_constant(cmd, pc);
    cmd.dispatch(1, 1, 1);
}

MedianApproxNode::NodeStatusFlags MedianApproxNode::properties(Properties& config) {
    config.config_options("component", component, {"R", "G", "B", "A"});

    config.config_float("min", pc.min);
    config.config_float("max", pc.max);

    return {};
}

} // namespace merian_nodes
