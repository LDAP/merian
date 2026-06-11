#pragma once

#include "merian-graph/connectors/image/vk_image_in.hpp"
#include "merian-graph/connectors/ptr_out.hpp"
#include "merian-graph/graph/node.hpp"

#include "merian/utils/input_controller.hpp"
#include "merian/vk/utils/blits.hpp"
#include "merian/vk/window/swapchain.hpp"
#include "merian/vk/window/swapchain_manager.hpp"
#include "merian/vk/window/window.hpp"
#include "merian/vk/window/window_provider.hpp"

namespace merian {

// Presents to a window from a WindowProvider extension (configurable via properties()).
class WindowNode : public Node {
  public:
    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags pre_process(const GraphRun& run, const NodeIO& io) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

    const SwapchainHandle& get_swapchain();

    const WindowHandle& get_window() const {
        return window;
    }

    // Configures the response to the window's close button: raise SIGINT and/or SIGTERM (for a host
    // that shuts down on those) and/or remove the node from the graph.
    void set_on_should_close(const bool sigint, const bool sigterm, const bool remove_node) {
        on_should_close_sigint = sigint;
        on_should_close_sigterm = sigterm;
        on_should_close_remove_node = remove_node;
    }

  private:
    const std::shared_ptr<WindowProvider>& get_selected_provider() const;

    uint32_t src_array_element = 0;
    uint32_t current_src_array_size = 1;

    ContextHandle context;
    std::vector<std::shared_ptr<WindowProvider>> providers;
    int selected_provider = 0;
    std::shared_ptr<WindowProvider> active_provider;

    WindowCreateInfo create_info;
    WindowHandle window;
    std::optional<SwapchainManager> swapchain_manager = std::nullopt;

    BlitMode mode = FIT;

    VkImageInHandle image_in = VkImageIn::transfer_src(0, true);

    PtrOutHandle<SwapchainAcquireResult> con_acquire = PtrOut<SwapchainAcquireResult>::create();
    PtrOutHandle<InputController> con_controller = PtrOut<InputController>::create();
    PtrOutHandle<Window> con_window = PtrOut<Window>::create();

    bool request_rebuild_on_recreate = false;
    uint64_t acquire_timeout_ns = 1000L * 1000L * 100L; // .1s

    bool on_should_close_sigint = false;
    bool on_should_close_sigterm = false;
    bool on_should_close_remove_node = true;

    bool throttle = false;
};

} // namespace merian
