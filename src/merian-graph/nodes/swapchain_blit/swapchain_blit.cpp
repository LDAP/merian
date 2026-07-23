#include "merian-graph/nodes/swapchain_blit/swapchain_blit.hpp"

namespace merian {

std::vector<InputConnectorDescriptor> SwapchainBlit::describe_inputs() {
    return {{"src", con_src, ConnectorAccess::transfer_src, 0, true}, {"acquire", con_acquire}};
}

std::vector<OutputConnectorDescriptor>
SwapchainBlit::describe_outputs(const NodeIOLayout& /*io_layout*/) {
    return {{"acquire", con_acquire_out}};
}

void SwapchainBlit::process(GraphRun& run, const NodeIO& io) {
    const std::shared_ptr<SwapchainAcquireResult>& acquire = io[con_acquire];
    io[con_acquire_out] = acquire;
    if (!acquire) {
        return;
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
        return;
    }

    // The acquired image's contents are undefined, so discard them on the transfer-dst barrier.
    const CommandBufferHandle& cmd = run.get_cmd();
    const ImageHandle dst_image = acquire->image_view->get_image();
    cmd->barrier(dst_image->barrier2(vk::ImageLayout::eTransferDstOptimal, true));

    const vk::Filter filter =
        src_image->format_features() & vk::FormatFeatureFlagBits::eSampledImageFilterLinear
            ? vk::Filter::eLinear
            : vk::Filter::eNearest;
    cmd_blit(mode, cmd, src_image, vk::ImageLayout::eGeneral, src_image->get_extent(), dst_image,
             vk::ImageLayout::eTransferDstOptimal, dst_image->get_extent(), vk::ClearColorValue{},
             filter);
}

Node::NodeStatusFlags SwapchainBlit::properties(Properties& config) {
    if (current_src_array_size > 0) {
        config.config_uint("source array element", src_array_element, "", 0U,
                           current_src_array_size - 1);
    }

    int int_mode = mode;
    config.config_options("blit mode", int_mode, {"FIT", "FILL", "STRETCH"},
                          Properties::OptionsStyle::LIST_BOX);
    mode = static_cast<BlitMode>(int_mode);

    return {};
}

} // namespace merian
