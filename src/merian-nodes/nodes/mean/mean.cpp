#include "merian-nodes/nodes/mean/mean.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "image_to_buffer.comp.spv.h"
#include "reduce_buffer.comp.spv.h"

#include "merian/shader/spriv_reflect.hpp"

namespace merian {

MeanToBuffer::MeanToBuffer() {}

MeanToBuffer::~MeanToBuffer() {}

DeviceSupportInfo MeanToBuffer::query_device_support(const DeviceSupportQueryInfo& query_info) {
    SpirvReflect reflect_i2b(merian_image_to_buffer_comp_spv(),
                             merian_image_to_buffer_comp_spv_size());
    SpirvReflect reflect_reduce(merian_reduce_buffer_comp_spv(),
                                merian_reduce_buffer_comp_spv_size());
    return reflect_i2b.query_device_support(query_info) &
           reflect_reduce.query_device_support(query_info);
}

void MeanToBuffer::initialize(const ContextHandle& context,
                              const ResourceAllocatorHandle& allocator) {
    Node::initialize(context, allocator);
    this->context = context;

    image_to_buffer_shader = EntryPoint::create(context, merian_image_to_buffer_comp_spv(),
                                                merian_image_to_buffer_comp_spv_size(), "main",
                                                vk::ShaderStageFlagBits::eCompute);
    reduce_buffer_shader = EntryPoint::create(context, merian_reduce_buffer_comp_spv(),
                                              merian_reduce_buffer_comp_spv_size(), "main",
                                              vk::ShaderStageFlagBits::eCompute);
}

std::vector<InputConnectorHandle> MeanToBuffer::describe_inputs() {
    return {con_src};
}

std::vector<OutputConnectorHandle> MeanToBuffer::describe_outputs(const NodeIOLayout& io_layout) {
    vk::Extent3D extent = io_layout[con_src]->get_create_info_or_throw().extent;

    const auto group_count_x = (extent.width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (extent.height + local_size_y - 1) / local_size_y;
    const std::size_t buffer_size = group_count_x * group_count_y;

    con_mean = std::make_shared<ManagedVkBufferOut>(
        "mean", vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::ShaderStageFlagBits::eCompute,
        vk::BufferCreateInfo({}, buffer_size * sizeof(merian::float4),
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
        uint32_t subgroup_size =
            context->get_physical_device()->get_properties().get_subgroup_properties().subgroupSize;
        auto image_to_buffer_spec_builder = SpecializationInfoBuilder();
        image_to_buffer_spec_builder.add_entry(local_size_x, local_size_y, subgroup_size);
        SpecializationInfoHandle spec = image_to_buffer_spec_builder.build();
        image_to_buffer = ComputePipeline::create(pipe_layout, image_to_buffer_shader, spec);

        auto reduce_buffer_spec_builder = SpecializationInfoBuilder();
        reduce_buffer_spec_builder.add_entry(local_size_x * local_size_y, 1, subgroup_size);
        spec = reduce_buffer_spec_builder.build();
        reduce_buffer = ComputePipeline::create(pipe_layout, reduce_buffer_shader, spec);
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

} // namespace merian
