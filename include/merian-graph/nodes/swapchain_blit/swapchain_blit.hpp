#pragma once

#include "merian-graph/connectors/image/vk_image_in.hpp"
#include "merian-graph/graph/node.hpp"

#include "merian/vk/utils/blits.hpp"

namespace merian {

// Blits a graph image ("src") onto a Window node's swapchain image ("canvas", written in place).
class SwapchainBlit : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    uint32_t src_array_element = 0;
    uint32_t current_src_array_size = 0;
    BlitMode mode = FIT;

    VkImageInHandle con_src = VkImageIn::create();
    VkImageInHandle con_canvas = VkImageIn::create();
};

} // namespace merian
