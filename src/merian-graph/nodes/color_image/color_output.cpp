#include "merian-graph/nodes/color_image/color_output.hpp"

namespace merian {

ColorImage::ColorImage() : Node() {}

ColorImage::~ColorImage() {}

std::vector<InputConnectorDescriptor> ColorImage::describe_inputs() {
    return {{"resolution", con_resolution, {}, 0, true}};
}

std::vector<OutputConnectorDescriptor>
ColorImage::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    needs_run = true;

    extent_from_input = io_layout.is_connected(con_resolution);
    if (extent_from_input) {
        extent = io_layout[con_resolution]->value();
    }

    con_out = ManagedVkImageOut::create(format, extent, true);

    return {{"out", con_out, ConnectorAccess::transfer_dst}};
}

void ColorImage::process([[maybe_unused]] GraphRun& run, const NodeIO& io) {
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

} // namespace merian
