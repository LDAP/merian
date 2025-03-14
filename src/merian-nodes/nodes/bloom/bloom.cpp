#include "merian-nodes/nodes/bloom/bloom.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "bloom_composite.comp.spv.h"
#include "bloom_separate.comp.spv.h"

namespace merian_nodes {

Bloom::Bloom(const ContextHandle& context) : Node(), context(context) {

    separate_module = std::make_shared<ShaderModule>(context, merian_bloom_separate_comp_spv_size(),
                                                     merian_bloom_separate_comp_spv());
    composite_module = std::make_shared<ShaderModule>(
        context, merian_bloom_composite_comp_spv_size(), merian_bloom_composite_comp_spv());
}

Bloom::~Bloom() {}

std::vector<InputConnectorHandle> Bloom::describe_inputs() {
    return {con_src};
}

std::vector<OutputConnectorHandle> Bloom::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::Format format = io_layout[con_src]->create_info.format;
    const vk::Extent3D extent = io_layout[con_src]->create_info.extent;

    con_out = ManagedVkImageOut::compute_write("out", format, extent);
    con_interm =
        ManagedVkImageOut::compute_read_write("interm", vk::Format::eR16G16B16A16Sfloat, extent);

    return {
        con_out,
        con_interm,
    };
}

Bloom::NodeStatusFlags Bloom::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                           const DescriptorSetLayoutHandle& descriptor_set_layout) {
    auto pipe_layout = PipelineLayoutBuilder(context)
                           .add_descriptor_set_layout(descriptor_set_layout)
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
                    const DescriptorSetHandle& descriptor_set,
                    const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();
    cmd->bind(separate);
    cmd->bind_descriptor_set(separate, descriptor_set);
    cmd->push_constant(separate, pc);
    cmd->dispatch(io[con_out]->get_extent(), local_size_x, local_size_y);

    const auto bar =
        io[con_interm]->barrier(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite,
                                vk::AccessFlagBits::eShaderRead);
    cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                 vk::PipelineStageFlagBits::eComputeShader, bar);

    cmd->bind(composite);
    cmd->bind_descriptor_set(composite, descriptor_set);
    cmd->push_constant(composite, pc);
    cmd->dispatch(io[con_out]->get_extent(), local_size_x, local_size_y);
}

Bloom::NodeStatusFlags Bloom::properties(Properties& config) {
    config.config_float("brightness threshold", pc.threshold,
                        "Only areas brighter than that are affected", .1);
    config.config_float("strengh", pc.strength, "Controls the strength of the effect", .0001);

    config.st_separate("Debug");
    bool value_changed =
        config.config_options("mode", mode, {"combined", "bloom only", "bloom off"});

    if (value_changed) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian_nodes
