#include "merian-nodes/nodes/mean/mean.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "image_to_buffer.comp.spv.h"
#include "reduce_buffer.comp.spv.h"

namespace merian_nodes {

MeanToBuffer::MeanToBuffer(const ContextHandle& context) : Node(), context(context) {

    image_to_buffer_shader = std::make_shared<ShaderModule>(
        context, merian_image_to_buffer_comp_spv_size(), merian_image_to_buffer_comp_spv());
    reduce_buffer_shader = std::make_shared<ShaderModule>(
        context, merian_reduce_buffer_comp_spv_size(), merian_reduce_buffer_comp_spv());
}

MeanToBuffer::~MeanToBuffer() {}

std::vector<InputConnectorHandle> MeanToBuffer::describe_inputs() {
    return {con_src};
}

std::vector<OutputConnectorHandle> MeanToBuffer::describe_outputs(const NodeIOLayout& io_layout) {
    vk::Extent3D extent = io_layout[con_src]->create_info.extent;

    const auto group_count_x = (extent.width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (extent.height + local_size_y - 1) / local_size_y;
    const std::size_t buffer_size = group_count_x * group_count_y;

    con_mean = std::make_shared<ManagedVkBufferOut>(
        "mean", vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo({}, buffer_size * sizeof(glm::vec4),
                             vk::BufferUsageFlagBits::eStorageBuffer));

    return {con_mean};
}

MeanToBuffer::NodeStatusFlags
MeanToBuffer::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                           const DescriptorSetLayoutHandle& descriptor_set_layout) {
    if (!image_to_buffer) {
        auto pipe_layout = PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(descriptor_set_layout)
                               .add_push_constant<PushConstant>()
                               .build_pipeline_layout();
        auto image_to_buffer_spec_builder = SpecializationInfoBuilder();
        image_to_buffer_spec_builder.add_entry(
            local_size_x, local_size_y,
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
        SpecializationInfoHandle spec = image_to_buffer_spec_builder.build();
        image_to_buffer =
            std::make_shared<ComputePipeline>(pipe_layout, image_to_buffer_shader, spec);

        auto reduce_buffer_spec_builder = SpecializationInfoBuilder();
        reduce_buffer_spec_builder.add_entry(
            local_size_x * local_size_y, 1,
            context->physical_device.physical_device_subgroup_properties.subgroupSize);
        spec = reduce_buffer_spec_builder.build();
        reduce_buffer = std::make_shared<ComputePipeline>(pipe_layout, reduce_buffer_shader, spec);
    }

    return {};
}

void MeanToBuffer::process([[maybe_unused]] GraphRun& run,
                           const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();
    const auto group_count_x = (io[con_src]->get_extent().width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (io[con_src]->get_extent().height + local_size_y - 1) / local_size_y;

    pc.divisor = io[con_src]->get_extent().width * io[con_src]->get_extent().height;

    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "image to buffer");
        cmd->bind(image_to_buffer);
        cmd->bind_descriptor_set(image_to_buffer, descriptor_set);
        cmd->push_constant(image_to_buffer, pc);
        cmd->dispatch(group_count_x, group_count_y, 1);
    }

    pc.size = group_count_x * group_count_y;
    pc.offset = 1;
    pc.count = group_count_x * group_count_y;

    while (pc.count > 1) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd,
                                 fmt::format("reduce {} elements", pc.count));
        auto bar = io[con_mean]->buffer_barrier(
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader, bar);

        cmd->bind(reduce_buffer);
        cmd->bind_descriptor_set(reduce_buffer, descriptor_set);
        cmd->push_constant(reduce_buffer, pc);
        cmd->dispatch((pc.count + workgroup_size - 1) / workgroup_size, 1, 1);

        pc.count = (pc.count + workgroup_size - 1) / workgroup_size;
        pc.offset *= workgroup_size;
    }
}

} // namespace merian_nodes
