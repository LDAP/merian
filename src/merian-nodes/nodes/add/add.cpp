#include "add.hpp"
#include "add.comp.spv.h"

#include "merian-nodes/graph/errors.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian_nodes {

Add::Add(const SharedContext context, const std::optional<vk::Format> output_format)
    : AbstractCompute(context), output_format(output_format) {
    shader =
        std::make_shared<ShaderModule>(context, merian_add_comp_spv_size(), merian_add_comp_spv());
}

Add::~Add() {}

std::vector<InputConnectorHandle> Add::describe_inputs() {
    return {
        con_a,
        con_b,
    };
}

std::vector<OutputConnectorHandle> Add::describe_outputs(const ConnectorIOMap& output_for_input) {
    vk::Format format;

    if (output_for_input.is_connected(con_a)) {
        extent = output_for_input[con_a]->create_info.extent;
        format = output_format.value_or(output_for_input[con_a]->create_info.format);
    } else if (output_for_input.is_connected(con_b)) {
        extent = output_for_input[con_b]->create_info.extent;
        format = output_format.value_or(output_for_input[con_b]->create_info.format);
    } else {
        throw graph_errors::node_error{"at least one input must be connected."};
    }

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_builder.add_entry<VkBool32>(output_for_input.is_connected(con_a));
    spec_builder.add_entry<VkBool32>(output_for_input.is_connected(con_b));
    spec_info = spec_builder.build();

    return {
        ManagedVkImageOut::compute_write("out", format, extent),
    };
}

SpecializationInfoHandle
Add::get_specialization_info([[maybe_unused]] const NodeIO& io) noexcept {
    return spec_info;
}

// const void* AddNode::get_push_constant([[maybe_unused]] GraphRun& run, [[maybe_unused]] const
// NodeIO& io) {
//     return &pc;
// }

std::tuple<uint32_t, uint32_t, uint32_t>
Add::get_group_count([[maybe_unused]] const merian_nodes::NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle Add::get_shader_module() {
    return shader;
}

Add::NodeStatusFlags Add::properties(Properties&) {
    return {};
}

} // namespace merian_nodes
