#pragma once

#include "merian-graph/connectors/image/vk_image_in.hpp"
#include "merian-graph/connectors/image/vk_image_out_external.hpp"
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
                                 const NodeIO& io,
                                 const NodeConnectionInfo& info,
                                 Submission& submission) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags pre_process(const NodeIO& io, const NodeProcessInfo& info) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

    // The optional input that is blitted to the swapchain.
    static constexpr std::string_view DISPLAY_INPUT = "src";

    const WindowHandle& get_window() const {
        return window;
    }

  private:
    const std::shared_ptr<WindowProvider>& get_selected_provider() const;

    const SwapchainHandle& get_swapchain();

    uint32_t src_array_element = 0;
    uint32_t current_src_array_size = 1;

    std::vector<std::shared_ptr<WindowProvider>> providers;
    // 0 is auto, then the providers
    int selected_backend = 0;
    std::shared_ptr<WindowProvider> active_provider;

    WindowHandle window;
    std::optional<SwapchainManager> swapchain_manager = std::nullopt;

    // empty names keep the swapchain's choice
    bool fullscreen = false;
    std::string surface_format;
    std::string present_mode;

    BlitMode mode = FIT;

    VkImageInHandle image_in = VkImageIn::create();

    // the acquired swapchain image, blitted from src or cleared; null when there is none
    ExternalVkImageOutHandle con_canvas = ExternalVkImageOut::create();
    PtrOutHandle<InputController> con_controller = PtrOut<InputController>::create();
    PtrOutHandle<Window> con_window = PtrOut<Window>::create();

    bool request_rebuild_on_recreate = false;
    uint64_t acquire_timeout_ns = 1000L * 1000L * 100L; // .1s

    bool on_close_send_event = false;
    bool on_close_remove_node = true;

    bool throttle = false;
};

} // namespace merian
