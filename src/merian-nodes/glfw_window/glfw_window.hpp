#pragma once

#include "merian-nodes/blit_external/blit_external.hpp"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/window/glfw_window.hpp"
#include "merian/vk/window/swapchain.hpp"

namespace merian {

template <BlitNodeMode mode = FIT> class GLFWWindowNode : public BlitExternalNode<mode> {
  public:
    GLFWWindowNode(const SharedContext context,
                   const GLFWWindowHandle window,
                   const SurfaceHandle surface,
                   const std::optional<QueueHandle> wait_queue = std::nullopt)
        : window(window), surface(surface) {
        swapchain = std::make_shared<merian::Swapchain>(context, surface, wait_queue);
        vsync = swapchain->vsync_enabled();
    }

    virtual std::string name() override {
        return "GLFW Window";
    }

    virtual void cmd_process(const vk::CommandBuffer& cmd,
                             GraphRun& run,
                             const uint32_t set_idx,
                             const std::vector<ImageHandle>& image_inputs,
                             const std::vector<BufferHandle>& buffer_inputs,
                             const std::vector<ImageHandle>& image_outputs,
                             const std::vector<BufferHandle>& buffer_outputs) override {

        swapchain->set_vsync(vsync);
        aquire = swapchain->aquire_auto_resize(*window);
        if (aquire) {
            BlitExternalNode<mode>::set_target(aquire->image, vk::ImageLayout::eUndefined,
                                               vk::ImageLayout::ePresentSrcKHR,
                                               vk::Extent3D(aquire->extent, 1));
            BlitExternalNode<mode>::cmd_process(cmd, run, set_idx, image_inputs, buffer_inputs,
                                                image_outputs, buffer_outputs);

            run.add_wait_semaphore(aquire->wait_semaphore, vk::PipelineStageFlagBits::eTransfer);
            run.add_signal_semaphore(aquire->signal_semaphore);
            run.add_submit_callback([&](const QueueHandle& queue) { swapchain->present(*queue); });
        }
    }

    SwapchainHandle get_swapchain() {
        return swapchain;
    }

    // allows to use the views before the run_callbacks call.
    std::optional<SwapchainAcquireResult>& current_aquire_result() {
        return aquire;
    }

    void get_configuration(Configuration& config, [[maybe_unused]] bool& needs_rebuild) override {

        GLFWmonitor* monitor = glfwGetWindowMonitor(*window);
        int fullscreen = monitor != NULL;
        const int old_fullscreen = fullscreen;
        config.config_options("mode", fullscreen, {"windowed", "fullscreen"});
        if (fullscreen != old_fullscreen) {
            if (fullscreen) {
                glfwGetWindowPos(*window, &windowed_pos_size[0], &windowed_pos_size[1]);
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

        // Perform the change in cmd_process, since recreating the swapchain here may interfere with
        // other accesses to the swapchain images.
        vsync = swapchain->vsync_enabled();
        config.config_bool("vsync", vsync, "Enables or disables vsync on the swapchain.");

        if (aquire) {
            config.output_text(fmt::format("surface format: {}\ncolor space: {}\nimage count: "
                                           "{}\nextent: {}x{}\npresent mode: {}",
                                           vk::to_string(aquire->surface_format.format),
                                           vk::to_string(aquire->surface_format.colorSpace),
                                           aquire->num_images, aquire->extent.width,
                                           aquire->extent.height,
                                           vk::to_string(swapchain->get_present_mode())));
        }
    }

  private:
    GLFWWindowHandle window;
    SurfaceHandle surface;
    SwapchainHandle swapchain;
    std::optional<SwapchainAcquireResult> aquire;

    std::array<int, 4> windowed_pos_size;
    bool vsync;
};

} // namespace merian
