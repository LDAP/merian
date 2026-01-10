#include "../../../../include/merian-nodes/nodes/blit/blit.hpp"

#include "merian/vk/utils/blits.hpp"

namespace merian_nodes {

Blit::Blit(const ContextHandle& context) {

}

std::vector<InputConnectorHandle> Blit::describe_inputs() {
    return {src_image_in, dst_image_in};
}

std::vector<OutputConnectorHandle> Blit::describe_outputs(const NodeIOLayout& io_layout) {
    return {};
}

void Blit::process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) {
    ImageHandle src_image;
    if (io.is_connected(src_image_in)) {
        current_src_array_size = io[src_image_in].get_array_size();
        src_array_element = std::min(src_array_element, current_src_array_size - 1);
        src_image = io[src_image_in].get_image(src_array_element);
    } else {
        current_src_array_size = 0;
    }

    ImageHandle dst_image;
    if (io.is_connected(dst_image_in) && io[dst_image_in].get_array_size() > 0) {
        src_image = io[dst_image_in].get_image(0);
    }

    const CommandBufferHandle& cmd = run.get_cmd();
    if (src_image && dst_image) {
        const vk::Filter filter =
            src_image->format_features() &
                    vk::FormatFeatureFlagBits::eSampledImageFilterLinear
                ? vk::Filter::eLinear
                : vk::Filter::eNearest;

        cmd_blit(mode, cmd, src_image, vk::ImageLayout::eTransferSrcOptimal,
                 src_image->get_extent(), dst_image, vk::ImageLayout::eTransferDstOptimal,
                 dst_image->get_extent(), vk::ClearColorValue{}, filter);

        run.add_pre_submit_callback([dst_image](const QueueHandle& queue, GraphRun& run) {
            const CommandBufferHandle& cmd = run.get_cmd();
            cmd->barrier(dst_image->barrier2(vk::ImageLayout::ePresentSrcKHR));
        });
    }
}

Node::NodeStatusFlags Blit::properties(Properties& config) {
    if (current_src_array_size > 0) {
        config.config_uint("source array element", src_array_element, 0,
                           current_src_array_size - 1);
    }

    int int_mode = mode;
    config.config_options("blit mode", int_mode, {"FIT", "FILL", "STRETCH"},
                          Properties::OptionsStyle::LIST_BOX);
    mode = (BlitMode)int_mode;

    return {};
}


}
