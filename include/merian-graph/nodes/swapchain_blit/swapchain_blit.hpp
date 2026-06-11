#pragma once

#include "merian-graph/connectors/image/vk_image_in.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/ptr_out.hpp"
#include "merian-graph/graph/node.hpp"

#include "merian/vk/utils/blits.hpp"
#include "merian/vk/window/swapchain_manager.hpp"

namespace merian {

// Blits a graph image ("src") onto a Window node's acquired swapchain image ("acquire"). The
// Window node owns acquisition and presentation; this node only records the blit.
class SwapchainBlit : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    uint32_t src_array_element = 0;
    uint32_t current_src_array_size = 0;
    BlitMode mode = FIT;

    VkImageInHandle con_src = VkImageIn::transfer_src(0, false);
    PtrInHandle<SwapchainAcquireResult> con_acquire = PtrIn<SwapchainAcquireResult>::create();
    // Passed through so further nodes (e.g. ImGui) chain after this blit and render on top.
    PtrOutHandle<SwapchainAcquireResult> con_acquire_out = PtrOut<SwapchainAcquireResult>::create();
};

} // namespace merian
