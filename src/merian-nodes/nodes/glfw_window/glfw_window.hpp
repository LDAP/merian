#pragma once

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/vk/extension/extension_vk_glfw.hpp"
#include "merian/vk/utils/barriers.hpp"
#include "merian/vk/utils/blits.hpp"
#include "merian/vk/window/glfw_window.hpp"
#include "merian/vk/window/swapchain.hpp"

namespace merian_nodes {

/*
 * Outputs to a GLFW window.
 * This node makes use of the error handling features of ExtensionVkGLFW, meaning, consider using
 * that to initialize GLFW.
 */
class GLFWWindow : public Node {
  public:
    GLFWWindow(const SharedContext context) : Node("GLFW window") {
        window = std::make_shared<merian::GLFWWindow>(context);
        surface = window->get_surface();
        swapchain = std::make_shared<merian::Swapchain>(context, surface);
        vsync = swapchain->vsync_enabled();
    }

    virtual std::vector<InputConnectorHandle> describe_inputs() override {
        return {image_in};
    }

    virtual void process(GraphRun& run,
                         const vk::CommandBuffer& cmd,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) override {
        auto& old_swapchains = io.frame_data<std::vector<SwapchainHandle>>();
        old_swapchains.clear();
        const auto& src_image = io[image_in];

        swapchain->set_vsync(vsync);

        acquire.reset();
        for (uint32_t tries = 0; !acquire && tries < 2; tries++) {
            try {
                acquire = swapchain->acquire(window, 1000 * 1000 /* 1s */);
            } catch (const Swapchain::needs_recreate& e) {
                old_swapchains.emplace_back(swapchain);
                swapchain = std::make_shared<Swapchain>(swapchain);
            }
        }

        if (acquire) {
            const auto bar = barrier_image_layout(acquire->image, vk::ImageLayout::eUndefined,
                                            vk::ImageLayout::eTransferDstOptimal,
                                            all_levels_and_layers(vk::ImageAspectFlagBits::eColor));
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eBottomOfPipe,
                                vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, bar);

            const vk::Filter filter =
                src_image->format_features() & vk::FormatFeatureFlagBits::eSampledImageFilterLinear
                    ? vk::Filter::eLinear
                    : vk::Filter::eNearest;
            const vk::Extent3D extent(acquire->extent, 1);

            cmd_blit(mode, cmd, *src_image, vk::ImageLayout::eTransferSrcOptimal,
                     src_image->get_extent(), acquire->image, vk::ImageLayout::eTransferDstOptimal,
                     extent, std::nullopt, filter);

            cmd_barrier_image_layout(cmd, acquire->image, vk::ImageLayout::eTransferDstOptimal,
                                     vk::ImageLayout::ePresentSrcKHR);

            on_blit_completed(cmd, *acquire);

            run.add_wait_semaphore(acquire->wait_semaphore, vk::PipelineStageFlagBits::eTransfer);
            run.add_signal_semaphore(acquire->signal_semaphore);
            run.add_submit_callback([&](const QueueHandle& queue) {
                try {
                    swapchain->present(queue);
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

    NodeStatusFlags configuration(Configuration& config) override {
        GLFWmonitor* monitor = glfwGetWindowMonitor(*window);
        int fullscreen = monitor != NULL;
        const int old_fullscreen = fullscreen;
        config.config_options("mode", fullscreen, {"windowed", "fullscreen"});
        if (fullscreen != old_fullscreen) {
            if (fullscreen) {
                try {
                    glfwGetWindowPos(*window, &windowed_pos_size[0], &windowed_pos_size[1]);
                } catch (const ExtensionVkGLFW::glfw_error& e) {
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
                              Configuration::OptionsStyle::LIST_BOX);
        mode = (BlitMode)int_mode;

        // Perform the change in cmd_process, since recreating the swapchain here may interfere
        // with other accesses to the swapchain images.
        vsync = swapchain->vsync_enabled();
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
    GLFWWindowHandle window;
    SurfaceHandle surface;

    SwapchainHandle swapchain;
    std::optional<SwapchainAcquireResult> acquire;
    BlitMode mode = FIT;

    std::function<void(const vk::CommandBuffer& cmd, SwapchainAcquireResult& acquire_result)>
        on_blit_completed = []([[maybe_unused]] const vk::CommandBuffer& cmd,
                               [[maybe_unused]] SwapchainAcquireResult& acquire_result) {};

    ManagedVkImageInHandle image_in = ManagedVkImageIn::transfer_src("src");

    std::array<int, 4> windowed_pos_size;
    bool vsync;
    bool request_rebuild_on_recreate = false;
};

} // namespace merian_nodes
