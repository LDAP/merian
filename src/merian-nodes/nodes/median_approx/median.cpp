#include "merian-nodes/nodes/median_approx/median.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "median_histogram.comp.spv.h"
#include "median_reduce.comp.spv.h"

namespace merian_nodes {

MedianApproxNode::MedianApproxNode(const ContextHandle& context) : Node(), context(context) {

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
MedianApproxNode::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {

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
MedianApproxNode::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                               const DescriptorSetLayoutHandle& descriptor_set_layout) {
    pipe_layout = PipelineLayoutBuilder(context)
                      .add_descriptor_set_layout(descriptor_set_layout)
                      .add_push_constant<PushConstant>()
                      .build_pipeline_layout();

    return {};
}

void MedianApproxNode::process([[maybe_unused]] GraphRun& run,
                               const CommandBufferHandle& cmd,
                               [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                               [[maybe_unused]] const NodeIO& io) {
    if (!pipe_reduce) {
        auto spec_builder = SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y, component);
        SpecializationInfoHandle spec = spec_builder.build();

        pipe_histogram = std::make_shared<ComputePipeline>(pipe_layout, histogram, spec);
        pipe_reduce = std::make_shared<ComputePipeline>(pipe_layout, reduce, spec);
    }

    cmd->fill(io[con_histogram], 0);
    auto bar = io[con_histogram]->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                 vk::AccessFlagBits::eShaderRead |
                                                     vk::AccessFlagBits::eShaderWrite);
    cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                 bar);

    cmd->bind(pipe_histogram);
    cmd->bind_descriptor_set(pipe_histogram, descriptor_set);
    cmd->push_constant(pipe_histogram, pc);
    cmd->dispatch(io[con_src]->get_extent(), local_size_x, local_size_y);

    bar = io[con_histogram]->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                vk::AccessFlagBits::eShaderWrite,
                                            vk::AccessFlagBits::eShaderRead);
    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                 vk::PipelineStageFlagBits::eComputeShader, bar);

    cmd->bind(pipe_reduce);
    cmd->bind_descriptor_set(pipe_reduce, descriptor_set);
    cmd->push_constant(pipe_reduce, pc);
    cmd->dispatch(1, 1, 1);
}

MedianApproxNode::NodeStatusFlags MedianApproxNode::properties(Properties& config) {
    if (config.config_options("component", component, {"R", "G", "B", "A"})) {
        pipe_reduce.reset();
    }

    config.config_float("min", pc.min);
    config.config_float("max", pc.max);

    return {};
}

} // namespace merian_nodes
