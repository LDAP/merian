#include "merian-graph/nodes/color_image/color_output.hpp"

namespace merian {

ColorImage::ColorImage() : Node() {}

ColorImage::~ColorImage() {}

std::vector<OutputConnectorDescriptor>
ColorImage::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    needs_run = true;

    con_out = ManagedVkImageOut::create(format, extent, true);

    return {{"out", con_out, ConnectorAccess::transfer_dst}};
}

[[nodiscard]] ColorImage::NodeStatusFlags ColorImage::process(
    const NodeIO& io, [[maybe_unused]] const NodeProcessInfo& info, Submission& submission) {
    if (needs_run) {
        submission.get_cmd()->clear(io[con_out], color);
        needs_run = false;
    }
    return {};
}

ColorImage::NodeStatusFlags ColorImage::properties(Properties& config) {
    needs_run = config.config_color4("color", (float*)&color);

    bool needs_reconnect = false;
    needs_reconnect |=
        config.config_enum("format", format, merian::Properties::OptionsStyle::COMBO);

    needs_reconnect |= config.config_uint("extent", &extent.width, "", 3);

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }

    return {};
}

} // namespace merian
