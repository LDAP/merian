#pragma once

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/vk/extension/extension_glfw.hpp"
#include "merian/vk/utils/barriers.hpp"
#include "merian/vk/utils/blits.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"
#include "merian/vk/window/glfw_window.hpp"
#include "merian/vk/window/swapchain.hpp"

namespace merian_nodes {

/*
 * Outputs to a GLFW window.
 * This node requires the error handling features of ExtensionVkGLFW
 */
class GLFWWindow : public Node {
  public:
    GLFWWindow(const ContextHandle& context) : Node() {
        if (context->get_extension<ExtensionGLFW>()) {
            window = std::make_shared<merian::GLFWWindow>(context);
            swapchain = std::make_shared<merian::Swapchain>(context, window->get_surface());
            vsync = swapchain->vsync_enabled();
        }
    }

    virtual std::vector<InputConnectorHandle> describe_inputs() override {
        if (!window) {
            throw graph_errors::node_error{"node requires ExtensionVkGLFW context extension"};
        }

        return {image_in};
    }

    virtual void process(GraphRun& run,
                         const vk::CommandBuffer& cmd,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) override {
        auto& old_swapchains = io.frame_data<std::vector<SwapchainHandle>>();
        old_swapchains.clear();

        vsync = swapchain->set_vsync(vsync);

        acquire.reset();
        for (uint32_t tries = 0; !acquire && tries < 2; tries++) {
            try {
                acquire = swapchain->acquire(window, 1000L * 1000L /* 1s */);
            } catch (const Swapchain::needs_recreate& e) {
                old_swapchains.emplace_back(swapchain);
                swapchain = std::make_shared<Swapchain>(swapchain);
            }
        }

        if (acquire) {
            const auto bar = barrier_image_layout(
                acquire->image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                all_levels_and_layers(vk::ImageAspectFlagBits::eColor));

            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eBottomOfPipe,
                                vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, bar);

            if (io.is_connected(image_in)) {
                const auto& src_image = io[image_in];
                const vk::Filter filter =
                    src_image->format_features() &
                            vk::FormatFeatureFlagBits::eSampledImageFilterLinear
                        ? vk::Filter::eLinear
                        : vk::Filter::eNearest;
                const vk::Extent3D extent(acquire->extent, 1);

                cmd_blit(mode, cmd, *src_image, vk::ImageLayout::eTransferSrcOptimal,
                         src_image->get_extent(), acquire->image,
                         vk::ImageLayout::eTransferDstOptimal, extent, vk::ClearColorValue{},
                         filter);
            } else {
                cmd.clearColorImage(acquire->image, vk::ImageLayout::eTransferDstOptimal,
                                    vk::ClearColorValue{}, all_levels_and_layers());
            }

            cmd_barrier_image_layout(cmd, acquire->image, vk::ImageLayout::eTransferDstOptimal,
                                     vk::ImageLayout::ePresentSrcKHR);

            on_blit_completed(cmd, *acquire);

            run.add_wait_semaphore(acquire->wait_semaphore, vk::PipelineStageFlagBits::eTransfer);
            run.add_signal_semaphore(acquire->signal_semaphore);
            run.add_submit_callback([&](const QueueHandle& queue, GraphRun& run) {
                try {
                    Stopwatch present_duration;
                    swapchain->present(queue);
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
        return swapchain;
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

        // Perform the change in cmd_process, since recreating the swapchain here may interfere
        // with other accesses to the swapchain images.
        config.config_bool("vsync", vsync, "Enables or disables vsync on the swapchain.");
        config.config_bool("rebuild on recreate", request_rebuild_on_recreate,
                           "requests a graph rebuild if the swapchain was recreated.");

        if (acquire) {
            config.output_text(fmt::format("surface format: {}\ncolor space: {}\nimage count: "
                                           "{}\nextent: {}x{}\npresent mode: {}",
                                           vk::to_string(acquire->surface_format.format),
                                           vk::to_string(acquire->surface_format.colorSpace),
                                           acquire->num_images, acquire->extent.width,
                                           acquire->extent.height,
                                           vk::to_string(swapchain->get_present_mode())));
        }
        return {};
    }

    // Window can be nullptr if GLFW extension is not available
    const GLFWWindowHandle& get_window() const {
        return window;
    }

    // Set a callback for when the blit of the node input was completed.
    // The image will have vk::ImageLayout::ePresentSrcKHR.
    void set_on_blit_completed(
        const std::function<void(const vk::CommandBuffer& cmd,
                                 SwapchainAcquireResult& acquire_result)>& on_blit_completed) {
        this->on_blit_completed = on_blit_completed;
    }

  private:
    GLFWWindowHandle window = nullptr;

    SwapchainHandle swapchain;
    std::optional<SwapchainAcquireResult> acquire;
    BlitMode mode = FIT;

    std::function<void(const vk::CommandBuffer& cmd, SwapchainAcquireResult& acquire_result)>
        on_blit_completed = []([[maybe_unused]] const vk::CommandBuffer& cmd,
                               [[maybe_unused]] SwapchainAcquireResult& acquire_result) {};

    ManagedVkImageInHandle image_in = ManagedVkImageIn::transfer_src("src", 0, true);

    std::array<int, 4> windowed_pos_size;
    bool vsync;
    bool request_rebuild_on_recreate = false;
};

} // namespace merian_nodes
