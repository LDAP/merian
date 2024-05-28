#include "color_output.hpp"

#include "merian/utils/glm.hpp"

namespace merian_nodes {

ColorImage::ColorImage(const vk::Format format,
                                 const vk::Extent3D extent,
                                 const vk::ClearColorValue color)
    : Node("Color Output"), color(color) {
    con_out = VkImageOut::transfer_write("out", format, extent, true);
}

ColorImage::~ColorImage() {}

std::vector<OutputConnectorHandle>
ColorImage::describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
    needs_run = true;

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

ColorImage::NodeStatusFlags ColorImage::configuration(Configuration& config) {
    const vk::ClearColorValue old_color = color;
    config.config_color("color", *merian::as_vec4((float*)&color));
    needs_run = old_color.float32 != color.float32;

    return {};
}

} // namespace merian_nodes
