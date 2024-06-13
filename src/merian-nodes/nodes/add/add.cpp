#include "add.hpp"
#include "add.comp.spv.h"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian_nodes {

Add::Add(const SharedContext context, const std::optional<vk::Format> output_format)
    : AbstractCompute(context, "Add"), output_format(output_format) {
    shader =
        std::make_shared<ShaderModule>(context, merian_add_comp_spv_size(), merian_add_comp_spv());

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info = spec_builder.build();
}

Add::~Add() {}

std::vector<InputConnectorHandle> Add::describe_inputs() {
    return {
        con_a,
        con_b,
    };
}

std::vector<OutputConnectorHandle>
Add::describe_outputs(const ConnectorIOMap& output_for_input) {
    extent = output_for_input[con_a]->create_info.extent;
    vk::Format format = output_format.value_or(output_for_input[con_a]->create_info.format);

    return {
        ManagedVkImageOut::compute_write("out", format, extent),
    };
}

SpecializationInfoHandle Add::get_specialization_info() const noexcept {
    return spec_info;
}

// const void* AddNode::get_push_constant([[maybe_unused]] GraphRun& run) {
//     return &pc;
// }

std::tuple<uint32_t, uint32_t, uint32_t> Add::get_group_count() const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle Add::get_shader_module() {
    return shader;
}

Add::NodeStatusFlags Add::configuration(Configuration&) {
    return {};
}

} // namespace merian_nodes
