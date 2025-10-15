#include "merian-nodes/nodes/color_image/color_output.hpp"

namespace merian_nodes {

ColorImage::ColorImage() : Node() {}

ColorImage::~ColorImage() {}

std::vector<InputConnectorHandle> ColorImage::describe_inputs() {
    return {con_resolution};
}

std::vector<OutputConnectorHandle>
ColorImage::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    needs_run = true;

    extent_from_input = io_layout.is_connected(con_resolution);
    if (extent_from_input) {
        extent = io_layout[con_resolution]->value();
    }

    con_out = ManagedVkImageOut::transfer_write("out", format, extent, true);

    return {con_out};
}

void ColorImage::process([[maybe_unused]] GraphRun& run,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) {
    if (needs_run) {
        run.get_cmd()->clear(io[con_out], color);
        needs_run = false;
    }
}

ColorImage::NodeStatusFlags ColorImage::properties(Properties& config) {
    needs_run = config.config_color4("color", (float*)&color);

    bool needs_reconnect = false;
    needs_reconnect |=
        config.config_enum("format", format, merian::Properties::OptionsStyle::COMBO);

    if (extent_from_input) {
        config.output_text("extent determined by input: {}x{}x{}", extent.width, extent.height,
                           extent.depth);
    } else {
        needs_reconnect |= config.config_uint("extent", &extent.width, "", 3);
    }

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }

    return {};
}

} // namespace merian_nodes
