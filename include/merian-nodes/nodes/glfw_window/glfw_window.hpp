#pragma once

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/vk/extension/extension_glfw.hpp"
#include "merian/vk/utils/blits.hpp"
#include "merian/vk/window/glfw_window.hpp"
#include "merian/vk/window/swapchain.hpp"
#include "merian/vk/window/swapchain_manager.hpp"

namespace merian_nodes {

/*
 * Outputs to a GLFW window.
 * This node requires the error handling features of ExtensionVkGLFW
 */
class GLFWWindow : public Node {
  public:
    GLFWWindow(const ContextHandle& context) : Node() {
        const auto glfw_ext = context->get_extension<ExtensionGLFW>();
        if (glfw_ext) {
            window = glfw_ext->create_window();

            const SwapchainHandle swapchain =
                std::make_shared<merian::Swapchain>(context, window->get_surface());
            swapchain_manager.emplace(swapchain);
        }
    }

    virtual std::vector<InputConnectorHandle> describe_inputs() override {
        if (!window) {
            throw graph_errors::node_error{"node requires ExtensionVkGLFW context extension"};
        }

        return {image_in};
    }

    virtual void process(GraphRun& run,
                         const CommandBufferHandle& cmd,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) override {
        const std::optional<SwapchainAcquireResult> acquire =
            swapchain_manager->acquire(window, 1000L * 1000L /* 1s */);

        if (acquire) {
            const ImageHandle image = acquire->image_view->get_image();

            auto barrier = image->barrier2(vk::ImageLayout::eTransferDstOptimal, true);
            cmd->barrier(barrier);

            if (io.is_connected(image_in)) {
                const auto& src_image = io[image_in];
                const vk::Filter filter =
                    src_image->format_features() &
                            vk::FormatFeatureFlagBits::eSampledImageFilterLinear
                        ? vk::Filter::eLinear
                        : vk::Filter::eNearest;

                cmd_blit(mode, cmd, src_image, vk::ImageLayout::eTransferSrcOptimal,
                         src_image->get_extent(), image, vk::ImageLayout::eTransferDstOptimal,
                         image->get_extent(), vk::ClearColorValue{}, filter);
            } else {
                cmd->clear(image);
            }

            cmd->barrier(image->barrier2(vk::ImageLayout::ePresentSrcKHR));

            on_blit_completed(cmd, *acquire);

            run.add_wait_semaphore(acquire->wait_semaphore, vk::PipelineStageFlagBits::eTransfer);
            run.add_signal_semaphore(acquire->signal_semaphore);

            uint32_t index = acquire->index;
            SwapchainHandle swapchain = get_swapchain();
            run.add_submit_callback([index, swapchain](const QueueHandle& queue, GraphRun& run) {
                try {
                    Stopwatch present_duration;
                    swapchain->present(queue, index);
                    run.hint_external_wait_time(present_duration.duration());
                } catch (const Swapchain::needs_recreate& e) {
                    // do nothing and hope for the best
                    return;
                }
            });

            if (request_rebuild_on_recreate && acquire->did_recreate)
                run.request_reconnect();
        }
    }

    const SwapchainHandle& get_swapchain() {
        return swapchain_manager->get_swapchain();
    }

    NodeStatusFlags properties(Properties& config) override {
        GLFWmonitor* monitor = window ? glfwGetWindowMonitor(*window) : nullptr;
        int fullscreen = static_cast<int>(monitor != nullptr);
        const int old_fullscreen = fullscreen;
        config.config_options("mode", fullscreen, {"windowed", "fullscreen"});
        if (window && fullscreen != old_fullscreen) {
            if (fullscreen != 0) {
                try {
                    glfwGetWindowPos(*window, &windowed_pos_size[0], &windowed_pos_size[1]);
                } catch (const ExtensionGLFW::glfw_error& e) {
                    if (e.id != GLFW_FEATURE_UNAVAILABLE) {
                        throw e;
                    }
                    windowed_pos_size[0] = windowed_pos_size[1] = 0;
                }
                glfwGetWindowSize(*window, &windowed_pos_size[2], &windowed_pos_size[3]);
                monitor = glfwGetPrimaryMonitor();
                const GLFWvidmode* vidmode = glfwGetVideoMode(monitor);
                glfwSetWindowMonitor(*window, monitor, 0, 0, vidmode->width, vidmode->height,
                                     vidmode->refreshRate);
            } else {
                glfwSetWindowMonitor(*window, NULL, windowed_pos_size[0], windowed_pos_size[1],
                                     windowed_pos_size[2], windowed_pos_size[3], GLFW_DONT_CARE);
            }
        }

        int int_mode = mode;
        config.config_options("blit mode", int_mode, {"FIT", "FILL", "STRETCH"},
                              Properties::OptionsStyle::LIST_BOX);
        mode = (BlitMode)int_mode;

        const SwapchainHandle swapchain = get_swapchain();

        // Perform the change in cmd_process, since recreating the swapchain here may interfere
        // with other accesses to the swapchain images.
        bool vsync = swapchain->vsync_enabled();
        if (config.config_bool("vsync", vsync, "Enables or disables vsync on the swapchain.")) {
            swapchain->set_vsync(vsync);
        }
        config.config_bool("rebuild on recreate", request_rebuild_on_recreate,
                           "requests a graph rebuild if the swapchain was recreated.");

        const std::optional<Swapchain::SwapchainInfo>& swapchain_info =
            swapchain->get_swapchain_info();

        if (swapchain_info) {
            config.output_text(fmt::format(
                "surface format: {}\ncolor space: {}\nimage count: "
                "{}\nextent: {}x{}\npresent mode: {}",
                vk::to_string(swapchain->get_surface_format().format),
                vk::to_string(swapchain->get_surface_format().colorSpace),
                swapchain_info->images.size(), swapchain_info->extent.width,
                swapchain_info->extent.height, vk::to_string(swapchain->get_present_mode())));
        }
        return {};
    }

    // Window can be nullptr if GLFW extension is not available
    const GLFWWindowHandle& get_window() const {
        return window;
    }

    // Set a callback for when the blit of the node input was completed.
    // The image will have vk::ImageLayout::ePresentSrcKHR.
    void
    set_on_blit_completed(const std::function<void(const CommandBufferHandle& cmd,
                                                   const SwapchainAcquireResult& acquire_result)>&
                              on_blit_completed) {
        this->on_blit_completed = on_blit_completed;
    }

  private:
    GLFWWindowHandle window = nullptr;
    std::optional<SwapchainManager> swapchain_manager;

    BlitMode mode = FIT;

    std::function<void(const CommandBufferHandle& cmd,
                       const SwapchainAcquireResult& acquire_result)>
        on_blit_completed = []([[maybe_unused]] const CommandBufferHandle& cmd,
                               [[maybe_unused]] const SwapchainAcquireResult& acquire_result) {};

    ManagedVkImageInHandle image_in = ManagedVkImageIn::transfer_src("src", 0, true);

    std::array<int, 4> windowed_pos_size;
    bool request_rebuild_on_recreate = false;
};

} // namespace merian_nodes
