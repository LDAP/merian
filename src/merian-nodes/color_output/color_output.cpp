#include "color_output.hpp"

namespace merian {

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
    status.skip_run = true;
}

void ColorOutputNode::cmd_build(const vk::CommandBuffer& cmd,
                                const std::vector<std::vector<merian::ImageHandle>>&,
                                const std::vector<std::vector<merian::BufferHandle>>&,
                                const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                                const std::vector<std::vector<merian::BufferHandle>>&) {
    const ImageHandle image = image_outputs[0][0];
    cmd.clearColorImage(*image, image->get_current_layout(), color, all_levels_and_layers());
}

} // namespace merian
