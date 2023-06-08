#pragma once

#include "merian/vk/command/queue_container.hpp"
#include "merian/vk/swapchain/surface.hpp"
#include "vulkan/vulkan.hpp"
#include <GLFW/glfw3.h>

#include <optional>

namespace merian {

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
    // Swapchain was created or recreated. You need to cmd_update_image_layouts().
    bool did_recreate;
    vk::Extent2D extent;
};

/**
 * @brief      This class describes a swapchain.
 *
 * Typical usage:
 *
 * auto result = swap.aquire_auto_resize();
 * if (!result.has_value) { handle }
 *
 * vk::CommandBuffer cmd = ...
 * if (result.value.did_recreate) {
 *   // after init or resize you have to setup the image layouts
 *   swap.cmd_update_image_layouts(cmd)
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
 * swap.present(queue);
 *
 */
class Swapchain : public std::enable_shared_from_this<Swapchain> {
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
    /**
     * @param[in]  preferred_surface_formats  The preferred surface formats in decreasing priority
     * @param[in]  fallback_format            The fallback format if non of the preferred formats is
     * available
     * @param[in]  wait_queue                 When recreating the swapchain it must be ensured that
     * all command buffers that have semaphores are processed. You can supply a queue to wait for.
     * If no queue is supplied, it is waited using device.waitIdle() (which is slower and not recommeded). 
     */
    Swapchain(const SharedContext& context,
              const SurfaceHandle& surface,
              const std::optional<QueueContainerHandle> wait_queue = std::nullopt,
              const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats =
                  {vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Srgb},
              const vk::PresentModeKHR preferred_vsync_off_mode = vk::PresentModeKHR::eMailbox);

    ~Swapchain();

    /* Never recreates the swapchain */
    std::optional<SwapchainAcquireResult> aquire() {
        return aquire_custom(cur_width, cur_height);
    }

    /* Recreates the swapchain if necessary according to window frame buffer size */
    std::optional<SwapchainAcquireResult> aquire_auto_resize(GLFWwindow* window) {
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

    void cmd_update_image_layouts(vk::CommandBuffer cmd) const {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, nullptr, barriers);
    }

    /* Remember to also transition image layouts */
    vk::Extent2D recreate_swapchain(int width, int height);

    /* Sets vsync and recreates the swapchain if necessary (without resize) */
    void set_vsync(bool state) {
        if (state != vsync) {
            present_mode = select_present_mode();
            recreate_swapchain(cur_width, cur_height);
        }
    }

  private:
    /* Destroys swapchain and image views */
    void destroy_swapchain();
    /* Destroys image views only (for recreate) */
    void destroy_entries();
    [[nodiscard]] vk::PresentModeKHR select_present_mode();
    void wait_idle();

  private:
    const SharedContext context;
    const SurfaceHandle surface;
    const std::vector<vk::SurfaceFormatKHR> preferred_surface_formats;
    const vk::PresentModeKHR preferred_vsync_off_mode;
    const std::optional<QueueContainerHandle> wait_queue;

    vk::SurfaceFormatKHR surface_format;
    bool vsync = false;
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

    vk::PresentModeKHR present_mode;
    // You should never access the swapchain directly
    vk::SwapchainKHR swapchain = VK_NULL_HANDLE;
};

using SwapchainHandle = std::shared_ptr<Swapchain>;

} // namespace merian
