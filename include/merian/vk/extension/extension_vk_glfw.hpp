#pragma once

#include "merian/utils/vector_utils.hpp"
#include "merian/vk/extension/extension.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

namespace merian {

/*
 * @brief      Initializes GLFW and manages swapchain, images and views as well as acquiring and
 * presenting.
 *
 * Typical usage:
 *
 * auto result = ext.aquire_auto_resize();
 * if (!result.has_value) { handle }
 *
 * vk::CommandBuffer cmd = ...
 * if (result.value.did_recreate) {
 *   // after init or resize you have to setup the image layouts
 *   ext.cmd_update_barriers(cmd)
 * }
 *
 * // render to result.imageView directly or own framebuffer then blit into the backbuffer
 * // cmd.blitImage(...result.image...)
 *
 * // Submit
 * vk::SubmitInfo submitInfo;
 *
 * // !! Important: Wait for the swapchain image to be read already!
 * VkPipelineStageFlags swapchainReadFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
 * submitInfo.waitSemaphoreCount = 1;
 * submitInfo.pWaitSemaphores    = &result.wait_semaphore;
 * submitInfo.pWaitDstStageMask  = &swapchainReadFlags);
 *
 * // !! After submit, signal write finished
 * submitInfo.signalSemaphoreCount = 1;
 * submitInfo.pSignalSemaphores    = result.signal_semaphore;
 *
 * queue.submit(1, &submitInfo, fence);
 * ext.present(queue);  // this extension makes sure that the graphics queue supports present
 *
 */
class ExtensionVkGLFW : public Extension {
  private:
    struct Entry {
        vk::Image image;
        vk::ImageView imageView;
    };
    struct SemaphoreGroup {
        // be aware semaphore index may not match active image index!
        vk::Semaphore read_semaphore;
        vk::Semaphore written_semaphore;
    };

  public:
    struct SwapchainAcquireResult {
        // The image and its view and index in the swap chain.
        vk::Image image;
        vk::ImageView view;
        uint32_t index;

        // You MUST wait on this semaphore before writing to the image. ("The
        // system" signals this semaphore when it's done presenting the
        // image and can safely be reused).
        vk::Semaphore wait_semaphore;
        // You MUST signal this semaphore when done writing to the image, and
        // before presenting it. (The system waits for this before presenting).
        vk::Semaphore signal_semaphore;
        // Swapchain was created or recreated. You may need to cmd_update_barriers().
        bool did_recreate;
        vk::Extent2D extent;
    };

  public:
    /**
     * @param[in]  preferred_surface_formats  The preferred surface formats in decreasing priority
     * @param[in]  fallback_format            The fallback format if non of the preferred formats is
     * available
     */
    ExtensionVkGLFW(int width = 1280,
                    int height = 720,
                    const char* title = "",
                    std::vector<vk::SurfaceFormatKHR> preferred_surface_formats =
                        {vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Srgb},
                    vk::PresentModeKHR preferred_vsync_off_mode = vk::PresentModeKHR::eMailbox)
        : Extension("ExtensionVkGLFW"), preferred_surface_formats(preferred_surface_formats),
          preferred_vsync_off_mode(preferred_vsync_off_mode) {
        if (!glfwInit())
            throw std::runtime_error("GLFW initialization failed!");
        if (!glfwVulkanSupported())
            throw std::runtime_error("GLFW reports to have no Vulkan support! Maybe it couldn't "
                                     "find the Vulkan loader!");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    }
    ~ExtensionVkGLFW() {
        glfwDestroyWindow(window);
    }
    std::vector<const char*> required_instance_extension_names() const override {
        std::vector<const char*> required_extensions;
        uint32_t count;
        const char** extensions = glfwGetRequiredInstanceExtensions(&count);
        required_extensions.insert(required_extensions.end(), extensions, extensions + count);
        return required_extensions;
    }
    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override {
        return {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };
    }
    void on_instance_created(const vk::Instance&) override;
    void on_destroy_instance(const vk::Instance&) override;
    bool accept_graphics_queue(const vk::PhysicalDevice&, std::size_t) override;
    void on_physical_device_selected(const Context::PhysicalDeviceContainer& pd_container) override;
    void on_device_created(const vk::Device&) override;
    void on_destroy_device(const vk::Device&) override;

  public: // own methods
    /* Sets vsync and recreates the swapchain if necessary (without resize) */
    void set_vsync(bool state) {
        if (state != vsync) {
            present_mode = select_present_mode();
            recreate_swapchain(cur_width, cur_height);
        }
    }
    /* Does not recreate the swapchain */
    std::optional<SwapchainAcquireResult> aquire() {
        return aquire_custom(cur_width, cur_height);
    }
    /* Recreates the swapchain if necessary according to window frame buffer size */
    std::optional<SwapchainAcquireResult> aquire_auto_resize() {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        return aquire_custom(width, height);
    }
    /* Recreates the swapchain if necessary */
    std::optional<SwapchainAcquireResult> aquire_custom(int width, int height);

    vk::Result present(vk::Queue queue);

    void present(QueueContainer& queue);

    /* Semaphore only valid until the next present() */
    vk::Semaphore& current_read_semaphore() {
        return semaphore_groups[current_semaphore_idx % semaphore_groups.size()].read_semaphore;
    }
    /* Semaphore only valid until the next present(). */
    vk::Semaphore& current_written_semaphore() {
        return semaphore_groups[current_semaphore_idx % semaphore_groups.size()].written_semaphore;
    }
    /* Image only valid until the next acquire_*() */
    vk::Image& current_image() {
        return entries[current_image_idx].image;
    }
    /* Image only valid until the next acquire_*() */
    vk::ImageView& current_image_view() {
        return entries[current_image_idx].imageView;
    }
    /* Image index only valid until the next acquire_*() */
    uint32_t current_image_index() {
        return current_image_idx;
    }
    uint32_t current_image_count() {
        return entries.size();
    }
    vk::ImageView image_view(uint32_t idx) const {
        check_size(entries, idx);
        return entries[idx].imageView;
    }
    vk::Image image(uint32_t idx) const {
        check_size(entries, idx);
        return entries[idx].image;
    }

    void cmd_update_barriers(vk::CommandBuffer cmd) const {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, nullptr, barriers);
    }
    /* Remember to also transition image layouts */
    vk::Extent2D recreate_swapchain(int width, int height);

  private:
    /* Destroys swapchain and image views */
    void destroy_swapchain();
    /* Destroys image views only (for recreate) */
    void destroy_entries();
    [[nodiscard]] vk::PresentModeKHR select_present_mode();

  private:
    std::vector<vk::SurfaceFormatKHR> preferred_surface_formats;
    vk::PresentModeKHR preferred_vsync_off_mode;
    bool vsync = false;

    vk::Device device = VK_NULL_HANDLE;
    vk::PhysicalDevice physical_device = VK_NULL_HANDLE;

    std::vector<Entry> entries;
    // updated in aquire_custom
    uint32_t current_image_idx;
    // updated in present
    std::vector<SemaphoreGroup> semaphore_groups;
    uint32_t current_semaphore_idx;
    std::vector<vk::ImageMemoryBarrier> barriers;
    int cur_width = 0;
    int cur_height = 0;
    // Only valid after the first acquire!
    vk::Extent2D extent;

  public:
    GLFWwindow* window;
    vk::SurfaceKHR surface;
    vk::SurfaceFormatKHR surface_format;
    vk::PresentModeKHR present_mode;
    // You should never access the swapchain directly
    vk::SwapchainKHR swapchain = VK_NULL_HANDLE;
};

} // namespace merian
