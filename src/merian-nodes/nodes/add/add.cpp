#include "merian-nodes/nodes/add/add.hpp"
#include "add.comp.spv.h"

#include "merian-nodes/graph/errors.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/math.hpp"

namespace merian_nodes {

Add::Add(const ContextHandle& context, const std::optional<vk::Format>& output_format)
    : AbstractCompute(context), output_format(output_format) {
    shader =
        std::make_shared<ShaderModule>(context, merian_add_comp_spv_size(), merian_add_comp_spv());
}

Add::~Add() {}

std::vector<InputConnectorHandle> Add::describe_inputs() {
    if (input_connectors.size() != number_inputs) {
        input_connectors.clear();
        for (uint32_t i = 0; i < number_inputs; i++) {
            input_connectors.emplace_back(
                ManagedVkImageIn::compute_read(fmt::format("input_{}", i), 0, true));
        }
    }

    return {input_connectors.begin(), input_connectors.end()};
}

std::vector<OutputConnectorHandle> Add::describe_outputs(const NodeIOLayout& io_layout) {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);

    bool at_least_one_input_connected = false;
    vk::Format format = output_format.value_or(vk::Format::eUndefined);
    extent = vk::Extent3D{
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
        std::numeric_limits<uint32_t>::max(),
    };

    for (const auto& input : input_connectors) {
        if (io_layout.is_connected(input)) {
            at_least_one_input_connected = true;
            if (format == vk::Format::eUndefined)
                format = io_layout[input]->create_info.format;
            extent = min(extent, io_layout[input]->create_info.extent);
        }
        spec_builder.add_entry(io_layout.is_connected(input));
    }

    if (!at_least_one_input_connected) {
        throw graph_errors::node_error{"at least one input must be connected."};
    }

    spec_info = spec_builder.build();
    return {
        ManagedVkImageOut::compute_write("out", format, extent),
    };
}

SpecializationInfoHandle Add::get_specialization_info([[maybe_unused]] const NodeIO& io) noexcept {
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

Add::NodeStatusFlags Add::properties(Properties& props) {
    props.output_text("output extent: {}x{}x{}", extent.width, extent.height, extent.depth);

    return {};
}

} // namespace merian_nodes
