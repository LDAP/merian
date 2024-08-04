#include "merian-nodes/nodes/color_image/color_output.hpp"

#include "merian/utils/glm.hpp"

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
                         const vk::CommandBuffer& cmd,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) {
    if (needs_run) {
        cmd.clearColorImage(*io[con_out], io[con_out]->get_current_layout(), color,
                            all_levels_and_layers());
        needs_run = false;
    }
}

ColorImage::NodeStatusFlags ColorImage::properties(Properties& config) {
    needs_run = config.config_color("color", *merian::as_vec4((float*)&color));

    bool needs_reconnect = false;
    needs_reconnect |=
        config.config_enum("format", format, merian::Properties::OptionsStyle::COMBO);

    if (extent_from_input) {
        config.output_text("extent determined by input: {}x{}x{}", extent.width, extent.height,
                           extent.depth);
    } else {
        needs_reconnect |= config.config_vec("extent", *merian::as_uvec3(&extent.width));
    }

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }

    return {};
}

} // namespace merian_nodes
