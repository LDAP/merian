#pragma once

#include "merian/vk/command/queue.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/sync/fence.hpp"
#include "merian/vk/sync/semaphore_binary.hpp"
#include "merian/vk/window/surface.hpp"

#include <optional>

namespace merian {

class Swapchain;
using SwapchainHandle = std::shared_ptr<Swapchain>;

class SwapchainImage : public Image {
  public:
    SwapchainImage(const ContextHandle& context,
                   const vk::Image& image,
                   const vk::ImageCreateInfo create_info,
                   const SwapchainHandle& swapchain);

    ~SwapchainImage() override;

  private:
    SwapchainHandle swapchain;
};

/**
 * @brief      This class describes a swapchain.
 */
class Swapchain : public std::enable_shared_from_this<Swapchain> {
    static constexpr uint32_t MAX_OLD_SWAPCHAIN_CHAIN_LENGTH = 5;

  public:
    struct SyncGroup {
        // be aware semaphore index may not match active image index!
        BinarySemaphoreHandle read_semaphore;
        BinarySemaphoreHandle written_semaphore;
        FenceHandle acquire_finished;
        bool acquire_in_progress = false;
    };

    class needs_recreate : public std::runtime_error {
      public:
        needs_recreate(const vk::Result& reason) : needs_recreate(vk::to_string(reason)) {}

        needs_recreate(const std::string& reason) : std::runtime_error(reason) {
            SPDLOG_DEBUG("needs recreate swapchain because {}", reason);
        }
    };

    struct SwapchainInfo {
        vk::ImageCreateInfo image_create_info; // image create info describing the swapchain images
        uint32_t min_images = 0;
        // Do not use directly. Use a SwapchainManager instead.
        std::vector<vk::Image> images;
        vk::Extent2D extent; // Only valid after the first acquire! (0,0) means swapchain is invalid
                             // and needs recreate.
        uint32_t cur_width = 0;
        uint32_t cur_height = 0;
        vk::PresentModeKHR cur_present_mode;
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
    //
    // Using this ensures the old swapchains are kept alive until all present operations have
    // finished. Also, it allows for some resource reuse.
    Swapchain(const SwapchainHandle& swapchain);

    ~Swapchain();

    // ---------------------------------------------------------------------------

    /* May throw Swapchain::needs_recreate.
     * For that you should use the Swapchains copy constructor.
     *
     * If the framebuffer extent is 0 or the aquire was not successfull, std::nullopt is returned.
     * Returns the swapchain image index and the sync group that must be used to sync access to the
     * swapchain images.
     */
    std::optional<std::pair<uint32_t, SyncGroup>> acquire(const vk::Extent2D extent,
                                                          const uint64_t timeout = UINT64_MAX);

    /* Transfers ownership of the image image_idx to the presentation engine for present.
     *
     * May throw Swapchain::needs_recreate.
     * For that you can use the Swapchains copy constructor.
     */
    void present(const QueueHandle& queue, const uint32_t image_idx);

    // ---------------------------------------------------------------------------

    vk::SurfaceFormatKHR get_surface_format() {
        return surface_format;
    }

    vk::PresentModeKHR get_present_mode() {
        return present_mode;
    }

    // needs at least one acquire to be valid. Also nullopt after needs_recreate was
    // thrown.
    const std::optional<SwapchainInfo>& get_swapchain_info() {
        return info;
    }

    const ContextHandle& get_context() const {
        return context;
    }

    /* Sets vsync. The swapchain is automatically recreated on next aquire.
     * Returns if vsync could be enabled.
     */
    bool set_vsync(bool state) {
        if (state != vsync_enabled()) {
            present_mode = select_present_mode(state);
        }
        return vsync_enabled();
    }

    // shortcut to check get_present_mode() == vk::PresentModeKHR::eFifo.
    bool vsync_enabled() const {
        return present_mode == vk::PresentModeKHR::eFifo;
    }

    // ---------------------------------------------------------------------------

  private:
    /* Remember to also transition image layouts */
    vk::Extent2D create_swapchain(const uint32_t width, const uint32_t height);

    [[nodiscard]] vk::PresentModeKHR select_present_mode(const bool vsync);

  private:
    const ContextHandle context;
    const SurfaceHandle surface;
    const std::vector<vk::SurfaceFormatKHR> preferred_surface_formats;
    const vk::PresentModeKHR preferred_vsync_off_mode;

    vk::SwapchainKHR swapchain = VK_NULL_HANDLE;

    vk::PresentModeKHR present_mode; // desired: Swapchain may throw needs_recreate to set.
    vk::SurfaceFormatKHR surface_format;

    // ---------------------------------------------------------------------------

    std::optional<SwapchainInfo> info;

    // ---------------------------------------------------------------------------
    // See https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/api/swapchain_recreation
    // we keep here a chain of old swapchains that are cleaned up when the next aquire is
    // successful.
    std::shared_ptr<Swapchain> old_swapchain;
    uint32_t old_swapchain_chain_length = 0;

    // if > num_images => Save to destroy old swapchain
    // since it means at least one present happend.
    // We then set save_to_destoy to true for the old swapchain, and reset the pointer.
    std::size_t acquire_count = 0;

    // set by the new swapchain, if false then a deviceIdle/queueIdle is necesarry when destroying.
    bool save_to_destoy = false;
    // ---------------------------------------------------------------------------

    // uint32_t current_sync_group_index = 0;  // = acquire_count % num_images;
    std::vector<SyncGroup> sync_groups;
    std::vector<uint32_t> image_idx_to_sync_group;
};

} // namespace merian
