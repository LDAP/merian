#include "merian-graph/nodes/swapchain_blit/swapchain_blit.hpp"

namespace merian {

std::vector<InputConnectorDescriptor> SwapchainBlit::describe_inputs() {
    return {{.name = "src",
             .connector = con_src,
             .access = ConnectorAccess::transfer_src,
             .optional = true},
            {.name = "canvas",
             .connector = con_canvas,
             .access = ConnectorAccess::transfer_dst,
             .in_place = true}};
}

[[nodiscard]] SwapchainBlit::NodeStatusFlags SwapchainBlit::process(
    const NodeIO& io, [[maybe_unused]] const NodeProcessInfo& info, Submission& submission) {
    const ImageHandle& dst_image = io[con_canvas].get_image();
    if (!dst_image) {
        return {};
    }

    ImageHandle src_image;
    if (io.is_connected(con_src)) {
        current_src_array_size = io[con_src].get_array_size();
        if (current_src_array_size > 0) {
            src_array_element = std::min(src_array_element, current_src_array_size - 1);
            src_image = io[con_src].get_image(src_array_element);
        }
    } else {
        current_src_array_size = 0;
    }

    if (!src_image) {
        return {};
    }

    // the blit covers the whole canvas
    const CommandBufferHandle& cmd = submission.get_cmd();
    cmd->barrier(dst_image->barrier2(vk::ImageLayout::eTransferDstOptimal, true));

    const vk::Filter filter =
        src_image->format_features() & vk::FormatFeatureFlagBits::eSampledImageFilterLinear
            ? vk::Filter::eLinear
            : vk::Filter::eNearest;
    cmd_blit(mode, cmd, src_image, vk::ImageLayout::eGeneral, src_image->get_extent(), dst_image,
             vk::ImageLayout::eTransferDstOptimal, dst_image->get_extent(), vk::ClearColorValue{},
             filter);
    return {};
}

Node::NodeStatusFlags SwapchainBlit::properties(Properties& config) {
    if (current_src_array_size > 0 || !config.is_ui()) {
        config.config_uint("source array element", src_array_element, "", 0U,
                           std::max(current_src_array_size, 1U) - 1);
    }

    int int_mode = mode;
    config.config_options("blit mode", int_mode, {"FIT", "FILL", "STRETCH"},
                          Properties::OptionsStyle::LIST_BOX);
    mode = static_cast<BlitMode>(int_mode);

    return {};
}

} // namespace merian
