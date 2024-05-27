#include "add.hpp"
#include "add.comp.spv.h"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian_nodes {

AddNode::AddNode(const SharedContext context, const std::optional<vk::Format> output_format)
    : ComputeNode(context, "Add"), output_format(output_format) {
    shader =
        std::make_shared<ShaderModule>(context, merian_add_comp_spv_size(), merian_add_comp_spv());

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info = spec_builder.build();
}

AddNode::~AddNode() {}

std::vector<InputConnectorHandle> AddNode::describe_inputs() {
    return {
        con_a,
        con_b,
    };
}

std::vector<OutputConnectorHandle>
AddNode::describe_outputs(const ConnectorIOMap& output_for_input) {
    extent = output_for_input[con_a]->create_info.extent;
    vk::Format format = output_format.value_or(output_for_input[con_a]->create_info.format);

    return {
        VkImageOut::compute_write("out", format, extent),
    };
}

SpecializationInfoHandle AddNode::get_specialization_info() const noexcept {
    return spec_info;
}

// const void* AddNode::get_push_constant([[maybe_unused]] GraphRun& run) {
//     return &pc;
// }

std::tuple<uint32_t, uint32_t, uint32_t> AddNode::get_group_count() const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle AddNode::get_shader_module() {
    return shader;
}

AddNode::NodeStatusFlags AddNode::configuration(Configuration&) {
    return {};
}

} // namespace merian_nodes
