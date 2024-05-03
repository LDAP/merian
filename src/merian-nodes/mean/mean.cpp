#include "mean.hpp"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "image_to_buffer.comp.spv.h"
#include "reduce_buffer.comp.spv.h"

namespace merian {

MeanNode::MeanNode(const SharedContext context, const ResourceAllocatorHandle allocator)
    : context(context), allocator(allocator) {

    image_to_buffer_shader = std::make_shared<ShaderModule>(
        context, merian_image_to_buffer_comp_spv_size(), merian_image_to_buffer_comp_spv());
    reduce_buffer_shader = std::make_shared<ShaderModule>(
        context, merian_reduce_buffer_comp_spv_size(), merian_reduce_buffer_comp_spv());
}

MeanNode::~MeanNode() {}

std::string MeanNode::name() {
    return "Mean";
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
MeanNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("src"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
MeanNode::describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                           const std::vector<NodeOutputDescriptorBuffer>&) {
    vk::Extent3D extent = connected_image_outputs[0].create_info.extent;

    const auto group_count_x = (extent.width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (extent.height + local_size_y - 1) / local_size_y;
    const std::size_t buffer_size = group_count_x * group_count_y;

    return {
        {},
        {
            NodeOutputDescriptorBuffer(
                "mean", vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::BufferCreateInfo({}, buffer_size * sizeof(glm::vec4),
                                     vk::BufferUsageFlagBits::eStorageBuffer)),
        },
    };
}

void MeanNode::cmd_build([[maybe_unused]] const vk::CommandBuffer& cmd,
                         const std::vector<NodeIO>& ios) {
    std::tie(graph_textures, graph_sets, graph_pool, graph_layout) =
        make_graph_descriptor_sets(context, allocator, ios, graph_layout);

    if (!image_to_buffer) {
        auto pipe_layout = PipelineLayoutBuilder(context)
                               .add_descriptor_set_layout(graph_layout)
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
}

void MeanNode::cmd_process(const vk::CommandBuffer& cmd,
                           GraphRun& run,
                           const std::shared_ptr<FrameData>& frame_data,
                           const uint32_t set_index,
                           const NodeIO& io) {
    const auto group_count_x =
        (io.image_inputs[0]->get_extent().width + local_size_x - 1) / local_size_x;
    const auto group_count_y =
        (io.image_inputs[0]->get_extent().height + local_size_y - 1) / local_size_y;

    pc.divisor = io.image_inputs[0]->get_extent().width * io.image_inputs[0]->get_extent().height;

    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "image to buffer");
        image_to_buffer->bind(cmd);
        image_to_buffer->bind_descriptor_set(cmd, graph_sets[set_index]);
        image_to_buffer->push_constant(cmd, pc);
        cmd.dispatch(group_count_x, group_count_y, 1);
    }

    pc.size = group_count_x * group_count_y;
    pc.offset = 1;
    pc.count = group_count_x * group_count_y;

    while (pc.count > 1) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd,
                                 fmt::format("reduce {} elements", pc.count));
        auto bar = io.buffer_outputs[0]->buffer_barrier(
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, bar, {});

        reduce_buffer->bind(cmd);
        reduce_buffer->bind_descriptor_set(cmd, graph_sets[set_index]);
        reduce_buffer->push_constant(cmd, pc);
        cmd.dispatch((pc.count + workgroup_size - 1) / workgroup_size, 1, 1);

        pc.count = (pc.count + workgroup_size - 1) / workgroup_size;
        pc.offset *= workgroup_size;
    }
}

void MeanNode::get_configuration(Configuration&, bool&) {}

} // namespace merian
