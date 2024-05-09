#include "bloom.hpp"
#include "merian-nodes/graph/graph.hpp"
#include "merian-nodes/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "bloom_composite.comp.spv.h"
#include "bloom_separate.comp.spv.h"

namespace merian {

BloomNode::BloomNode(const SharedContext context, const ResourceAllocatorHandle allocator)
    : context(context), allocator(allocator) {

    separate_module = std::make_shared<ShaderModule>(context, merian_bloom_separate_comp_spv_size(),
                                                     merian_bloom_separate_comp_spv());
    composite_module = std::make_shared<ShaderModule>(
        context, merian_bloom_composite_comp_spv_size(), merian_bloom_composite_comp_spv());
}

BloomNode::~BloomNode() {}

std::string BloomNode::name() {
    return "Bloom";
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
BloomNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("src"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
BloomNode::describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                            const std::vector<NodeOutputDescriptorBuffer>&) {
    vk::Format format = connected_image_outputs[0].create_info.format;
    vk::Extent3D extent = connected_image_outputs[0].create_info.extent;
    return {
        {
            NodeOutputDescriptorImage::compute_write("output", format, extent),
            NodeOutputDescriptorImage::compute_read_write("interm", vk::Format::eR16G16B16A16Sfloat,
                                                          extent),
        },
        {},
    };
}

void BloomNode::cmd_build([[maybe_unused]] const vk::CommandBuffer& cmd,
                          const std::vector<NodeIO>& ios) {
    std::tie(graph_textures, graph_sets, graph_pool, graph_layout) =
        make_graph_descriptor_sets(context, allocator, ios, graph_layout);

    auto pipe_layout = PipelineLayoutBuilder(context)
                           .add_descriptor_set_layout(graph_layout)
                           .add_push_constant<PushConstant>()
                           .build_pipeline_layout();
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, mode);
    SpecializationInfoHandle spec = spec_builder.build();

    separate = std::make_shared<ComputePipeline>(pipe_layout, separate_module, spec);
    composite = std::make_shared<ComputePipeline>(pipe_layout, composite_module, spec);
}

void BloomNode::cmd_process(const vk::CommandBuffer& cmd,
                            [[maybe_unused]] GraphRun& run,
                            [[maybe_unused]] const std::shared_ptr<FrameData>& frame_data,
                            const uint32_t set_index,
                            const NodeIO& io) {
    const auto group_count_x =
        (io.image_outputs[0]->get_extent().width + local_size_x - 1) / local_size_x;
    const auto group_count_y =
        (io.image_outputs[0]->get_extent().height + local_size_y - 1) / local_size_y;

    separate->bind(cmd);
    separate->bind_descriptor_set(cmd, graph_sets[set_index]);
    separate->push_constant(cmd, pc);
    cmd.dispatch(group_count_x, group_count_y, 1);

    auto bar =
        io.image_outputs[1]->barrier(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite,
                                     vk::AccessFlagBits::eShaderRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, bar);

    composite->bind(cmd);
    composite->bind_descriptor_set(cmd, graph_sets[set_index]);
    composite->push_constant(cmd, pc);
    cmd.dispatch(group_count_x, group_count_y, 1);
}

void BloomNode::get_configuration(Configuration& config, bool& needs_rebuild) {
    config.config_float("brightness threshold", pc.threshold,
                        "Only areas brighter than that are affected", .1);
    config.config_float("strengh", pc.strength, "Controls the strength of the effect", .0001);

    config.st_separate("Debug");
    int32_t old_mode = mode;
    config.config_options("mode", mode, {"combined", "bloom only", "bloom off"});
    needs_rebuild |= old_mode != mode;
}

} // namespace merian
