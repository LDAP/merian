#pragma once

#include "merian/vk/command/queue.hpp"
#include "merian/vk/sync/semaphore_binary.hpp"
#include "merian/vk/window/surface.hpp"
#include "merian/vk/window/window.hpp"
#include "vulkan/vulkan.hpp"
#include <GLFW/glfw3.h>

#include <optional>

namespace merian {

class Swapchain;
using SwapchainHandle = std::shared_ptr<Swapchain>;

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
    // A pointer to the old swapchain if it was not already destroyed.
    // Can be used to enqueue cleanup functions.
    std::weak_ptr<Swapchain> old_swapchain;

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
 *   // after init or resize you have can use cmd_update_image_layouts to setup the image layouts
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

  public:
    class needs_recreate : public std::runtime_error {
      public:
        needs_recreate(const vk::Result& reason) : needs_recreate(vk::to_string(reason)) {}

        needs_recreate(const std::string& reason) : std::runtime_error(reason) {
            SPDLOG_DEBUG("needs recreate swapchain because {}", reason);
        }
    };

    /**
     * @param[in]  preferred_surface_formats  The preferred surface formats in decreasing priority
     * @param[in]  fallback_format            The fallback format if non of the preferred formats is
     * available
     */
    Swapchain(const ContextHandle& context,
              const SurfaceHandle& surface,
              const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats =
                  {vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Srgb},
              const vk::PresentModeKHR preferred_vsync_off_mode = vk::PresentModeKHR::eMailbox);

    // Special constructor that recreates the swapchain.
    // The old is then not usable anymore but must be kept alive until the old images finished
    // any in-flight processing (fancy by keeping alive until the current frame finished on the GPU
    // or just wait_idle on the queue or device).
    Swapchain(const SwapchainHandle& swapchain);

    ~Swapchain();

    /* May throw Swapchain::needs_recreate if the swapchain needs to adapt to a new to window frame
     * buffer size For that you can use the Swapchains copy constructor.
     *
     * If the framebuffer extent is 0 or the aquire was not successfull, std::nullopt is returned.
     */
    std::optional<SwapchainAcquireResult> acquire(const WindowHandle& window,
                                                  const uint64_t timeout = UINT64_MAX) {
        return acquire([&]() { return window->framebuffer_extent(); }, timeout);
    }

    /* May throw Swapchain::needs_recreate.
     * For that you can use the Swapchains copy constructor.
     *
     * If the framebuffer extent is 0 or the aquire was not successfull, std::nullopt is returned.
     */
    std::optional<SwapchainAcquireResult>
    acquire(const std::function<vk::Extent2D()>& framebuffer_extent,
            const uint64_t timeout = UINT64_MAX);

    /* May throw Swapchain::needs_recreate.
     * For that you can use the Swapchains copy constructor.
     */
    void present(const QueueHandle& queue);

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
    uint32_t current_image_index() const {
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
    vk::Extent2D create_swapchain(const uint32_t width, const uint32_t height);

    /* Sets vsync. The swapchain is automatically recreated on next aquire.
     * Returns if vsync could be enabled.
     */
    bool set_vsync(bool state) {
        if (state != vsync_enabled()) {
            present_mode = select_present_mode(state);
        }
        return vsync_enabled();
    }

    bool vsync_enabled() const {
        return present_mode == vk::PresentModeKHR::eFifo;
    }

    vk::PresentModeKHR get_present_mode() {
        return present_mode;
    }

    // intened to destroy framebuffers and renderpasses when the swapchain is destroyed.
    void add_cleanup_function(const std::function<void()>& cleanup_function) {
        cleanup_functions.emplace_back(cleanup_function);
    }

  private:
    /* Destroys swapchain and image views */
    void destroy_swapchain();

    /* Destroys image views only (for recreate) */
    void destroy_entries();
    [[nodiscard]] vk::PresentModeKHR select_present_mode(const bool vsync);

  private:
    const ContextHandle context;
    const SurfaceHandle surface;
    const std::vector<vk::SurfaceFormatKHR> preferred_surface_formats;
    const vk::PresentModeKHR preferred_vsync_off_mode;
    const std::optional<QueueHandle> wait_queue;

    std::vector<std::function<void()>> cleanup_functions;

    uint32_t min_images = 0;
    uint32_t num_images = 0;

    vk::SurfaceFormatKHR surface_format;
    std::vector<Entry> entries;
    // updated in aquire_custom
    uint32_t current_image_idx;
    // updated in present
    std::vector<SemaphoreGroup> semaphore_groups;
    uint32_t current_semaphore_idx = 0;
    std::vector<vk::ImageMemoryBarrier> barriers;
    uint32_t cur_width = 0;
    uint32_t cur_height = 0;
    // Only valid after the first acquire!
    vk::Extent2D extent;

    // swapchain need recreate if this does not match cur_present_mode
    vk::PresentModeKHR present_mode;
    vk::PresentModeKHR cur_present_mode;

    vk::SwapchainKHR swapchain = VK_NULL_HANDLE;
    std::weak_ptr<Swapchain> old_swapchain;

    bool created = true;
};

} // namespace merian
