#include "bloom.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "bloom_composite.comp.spv.h"
#include "bloom_separate.comp.spv.h"

namespace merian_nodes {

Bloom::Bloom(const SharedContext context) : Node("Bloom"), context(context) {

    separate_module = std::make_shared<ShaderModule>(context, merian_bloom_separate_comp_spv_size(),
                                                     merian_bloom_separate_comp_spv());
    composite_module = std::make_shared<ShaderModule>(
        context, merian_bloom_composite_comp_spv_size(), merian_bloom_composite_comp_spv());
}

Bloom::~Bloom() {}

std::vector<InputConnectorHandle> Bloom::describe_inputs() {
    return {con_src};
}

std::vector<OutputConnectorHandle>
Bloom::describe_outputs(const ConnectorIOMap& output_for_input) {
    const vk::Format format = output_for_input[con_src]->create_info.format;
    const vk::Extent3D extent = output_for_input[con_src]->create_info.extent;

    con_out = ManagedVkImageOut::compute_write("out", format, extent);
    con_interm = ManagedVkImageOut::compute_read_write("interm", vk::Format::eR16G16B16A16Sfloat, extent);

    return {
        con_out,
        con_interm,
    };
}

Bloom::NodeStatusFlags Bloom::on_connected(const DescriptorSetLayoutHandle& graph_layout) {
    auto pipe_layout = PipelineLayoutBuilder(context)
                           .add_descriptor_set_layout(graph_layout)
                           .add_push_constant<PushConstant>()
                           .build_pipeline_layout();
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, mode);
    SpecializationInfoHandle spec = spec_builder.build();

    separate = std::make_shared<ComputePipeline>(pipe_layout, separate_module, spec);
    composite = std::make_shared<ComputePipeline>(pipe_layout, composite_module, spec);

    return {};
}

void Bloom::process([[maybe_unused]] GraphRun& run,
                        const vk::CommandBuffer& cmd,
                        const DescriptorSetHandle& descriptor_set,
                        const NodeIO& io) {
    const auto group_count_x = (io[con_out]->get_extent().width + local_size_x - 1) / local_size_x;
    const auto group_count_y = (io[con_out]->get_extent().height + local_size_y - 1) / local_size_y;

    separate->bind(cmd);
    separate->bind_descriptor_set(cmd, descriptor_set);
    separate->push_constant(cmd, pc);
    cmd.dispatch(group_count_x, group_count_y, 1);

    auto bar = io[con_interm]->barrier(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite,
                                       vk::AccessFlagBits::eShaderRead);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, bar);

    composite->bind(cmd);
    composite->bind_descriptor_set(cmd, descriptor_set);
    composite->push_constant(cmd, pc);
    cmd.dispatch(group_count_x, group_count_y, 1);
}

Bloom::NodeStatusFlags Bloom::properties(Properties& config) {
    config.config_float("brightness threshold", pc.threshold,
                        "Only areas brighter than that are affected", .1);
    config.config_float("strengh", pc.strength, "Controls the strength of the effect", .0001);

    config.st_separate("Debug");
    int32_t old_mode = mode;
    config.config_options("mode", mode, {"combined", "bloom only", "bloom off"});

    if (old_mode != mode) {
        return NEEDS_RECONNECT;
    } else {
        return {};
    }
}

} // namespace merian_nodes
