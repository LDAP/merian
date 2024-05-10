#pragma once

#include "merian/vk/command/queue.hpp"
#include "merian/vk/window/surface.hpp"
#include "merian/vk/window/window.hpp"
#include "merian/vk/sync/semaphore_binary.hpp"
#include "vulkan/vulkan.hpp"
#include <GLFW/glfw3.h>

#include <optional>

namespace merian {

struct SwapchainAcquireResult {
    // The image and its view and index in the swap chain.
    vk::Image image;
    vk::ImageView view;
    uint32_t index;

    uint16_t num_images;
    uint16_t min_images;
    vk::SurfaceFormatKHR surface_format;

    // You MUST wait on this semaphore before writing to the image. ("The
    // system" signals this semaphore when it's done presenting the
    // image and can safely be reused).
    BinarySemaphoreHandle wait_semaphore;
    // You MUST signal this semaphore when done writing to the image, and
    // before presenting it. (The system waits for this before presenting).
    BinarySemaphoreHandle signal_semaphore;
    // Swapchain was created or recreated.
    // You can use cmd_update_image_layouts() to update the image layouts to PresentSrc.
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
        vk::Image image{};
        vk::ImageView imageView{};
    };

    struct SemaphoreGroup {
        // be aware semaphore index may not match active image index!
        BinarySemaphoreHandle read_semaphore{};
        BinarySemaphoreHandle written_semaphore{};
    };

    // Number of retries if aquire or present failes with Suboptimal/OutOfDate.
    static constexpr uint32_t ERROR_RETRIES = 2;

  public:
    /**
     * @param[in]  preferred_surface_formats  The preferred surface formats in decreasing priority
     * @param[in]  fallback_format            The fallback format if non of the preferred formats is
     * available
     * @param[in]  wait_queue                 When recreating the swapchain it must be ensured that
     * all command buffers that have semaphores are processed. You can supply a queue to wait for.
     * If no queue is supplied, it is waited using device.waitIdle() (which is slower and not
     * recommeded).
     */
    Swapchain(const SharedContext& context,
              const SurfaceHandle& surface,
              const std::optional<QueueHandle> wait_queue = std::nullopt,
              const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats =
                  {vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Srgb},
              const vk::PresentModeKHR preferred_vsync_off_mode = vk::PresentModeKHR::eMailbox);

    ~Swapchain();

    /* Recreates the swapchain if necessary according to window frame buffer size */
    std::optional<SwapchainAcquireResult> acquire(const WindowHandle& window) {
        return acquire([&]() { return window->framebuffer_extent(); });
    }

    /* Recreates the swapchain if necessary */
    std::optional<SwapchainAcquireResult>
    acquire(const std::function<vk::Extent2D()>& framebuffer_extent);

    void present(const QueueHandle& queue, const WindowHandle& window) {
        present(queue, [&]() { return window->framebuffer_extent(); });
    }

    void present(const QueueHandle& queue, const std::function<vk::Extent2D()>& framebuffer_extent);

    /* Semaphore only valid until the next present() */
    const BinarySemaphoreHandle& current_read_semaphore() const {
        return semaphore_groups[current_semaphore_idx % semaphore_groups.size()].read_semaphore;
    }

    /* Semaphore only valid until the next present(). */
    const BinarySemaphoreHandle& current_written_semaphore() const {
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

    vk::ImageView image_view(uint32_t idx) const;

    vk::Image image(uint32_t idx) const;

    vk::SurfaceFormatKHR get_surface_format() {
        return surface_format;
    }

    void cmd_update_image_layouts(vk::CommandBuffer cmd) const {
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, nullptr, barriers);
    }

    /* Remember to also transition image layouts */
    vk::Extent2D recreate_swapchain(int width, int height);

    /* Sets vsync and recreates the swapchain if necessary (without resize) */
    void set_vsync(bool state) {
        if (state != vsync_enabled()) {
            present_mode = select_present_mode(state);
        }
    }

    bool vsync_enabled() const {
        return cur_present_mode == vk::PresentModeKHR::eFifo;
    }

    vk::PresentModeKHR get_present_mode() {
        return cur_present_mode;
    }

  private:
    /* Destroys swapchain and image views */
    void destroy_swapchain();
    /* Destroys image views only (for recreate) */
    void destroy_entries();
    [[nodiscard]] vk::PresentModeKHR select_present_mode(const bool vsync);
    void wait_idle();

  private:
    const SharedContext context;
    const SurfaceHandle surface;
    const std::vector<vk::SurfaceFormatKHR> preferred_surface_formats;
    const vk::PresentModeKHR preferred_vsync_off_mode;
    const std::optional<QueueHandle> wait_queue;

    uint32_t min_images = 0;
    uint32_t num_images = 0;

    vk::SurfaceFormatKHR surface_format;
    std::vector<Entry> entries;
    // updated in aquire_custom
    uint32_t current_image_idx;
    // updated in present
    std::vector<SemaphoreGroup> semaphore_groups;
    uint32_t current_semaphore_idx;
    std::vector<vk::ImageMemoryBarrier> barriers;
    uint32_t cur_width = 0;
    uint32_t cur_height = 0;
    // Only valid after the first acquire!
    vk::Extent2D extent;

    // swapchain is recreated if this does not match cur_present_mode
    vk::PresentModeKHR present_mode;
    vk::PresentModeKHR cur_present_mode;
    // You should never access the swapchain directly
    vk::SwapchainKHR swapchain = VK_NULL_HANDLE;
    bool recreated = true;
};

using SwapchainHandle = std::shared_ptr<Swapchain>;

} // namespace merian
