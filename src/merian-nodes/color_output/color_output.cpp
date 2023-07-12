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

void ColorOutputNode::pre_process(NodeStatus& status) {
    status.skip_run = needs_run;
}

void ColorOutputNode::cmd_build(const vk::CommandBuffer& cmd,
                                const std::vector<std::vector<ImageHandle>>&,
                                const std::vector<std::vector<BufferHandle>>&,
                                const std::vector<std::vector<ImageHandle>>& image_outputs,
                                const std::vector<std::vector<BufferHandle>>&) {
    cmd.clearColorImage(*image_outputs[0][0], image_outputs[0][0]->get_current_layout(), color,
                        all_levels_and_layers());
}

void ColorOutputNode::cmd_process(const vk::CommandBuffer& cmd,
                                  GraphRun&,
                                  const uint32_t,
                                  const std::vector<ImageHandle>&,
                                  const std::vector<BufferHandle>&,
                                  const std::vector<ImageHandle>& image_outputs,
                                  const std::vector<BufferHandle>&) {
    cmd.clearColorImage(*image_outputs[0], image_outputs[0]->get_current_layout(), color,
                        all_levels_and_layers());
    needs_run = false;
}

void ColorOutputNode::get_configuration(Configuration& config) {
    const vk::ClearColorValue old_color = color;
    config.config_color("color", *merian::as_vec4((float*)&color));
    needs_run = old_color.float32 != color.float32;
}

} // namespace merian
