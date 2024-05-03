#include "color_output.hpp"
#include "merian/utils/glm.hpp"

namespace merian {

ColorOutputNode::ColorOutputNode(const vk::Format format,
                                 const vk::Extent3D extent,
                                 const vk::ClearColorValue color)
    : format(format), color(color), extent(extent) {}

ColorOutputNode::~ColorOutputNode() {}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
ColorOutputNode::describe_outputs(const std::vector<NodeOutputDescriptorImage>&,
                                  const std::vector<NodeOutputDescriptorBuffer>&) {
    return {
        {
            NodeOutputDescriptorImage::transfer_write("output", format, extent, true),
        },
        {},
    };
}

void ColorOutputNode::pre_process([[maybe_unused]] const uint64_t& iteration, NodeStatus& status) {
    status.skip_run = needs_run;
}

void ColorOutputNode::cmd_build(const vk::CommandBuffer& cmd, const std::vector<NodeIO>& ios) {
    cmd.clearColorImage(*ios[0].image_outputs[0], ios[0].image_outputs[0]->get_current_layout(),
                        color, all_levels_and_layers());
}

void ColorOutputNode::cmd_process(const vk::CommandBuffer& cmd,
                                  [[maybe_unused]] GraphRun& run,
                                  [[maybe_unused]] const std::shared_ptr<FrameData>& frame_data,
                                  [[maybe_unused]] const uint32_t set_index,
                                  const NodeIO& io) {
    cmd.clearColorImage(*io.image_outputs[0], io.image_outputs[0]->get_current_layout(), color,
                        all_levels_and_layers());
    needs_run = false;
}

void ColorOutputNode::get_configuration(Configuration& config, bool&) {
    const vk::ClearColorValue old_color = color;
    config.config_color("color", *merian::as_vec4((float*)&color));
    needs_run = old_color.float32 != color.float32;
}

} // namespace merian
