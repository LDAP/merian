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

        uint64_t number_acquires;
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

        vk::PresentModeKHR present_mode;
        vk::SurfaceFormatKHR surface_format;
    };

    /**
     * @param[in]  preferred_surface_formats  The preferred surface formats in decreasing priority
     * @param[in]  fallback_format            The fallback format if non of the preferred formats is
     * available
     */
    Swapchain(const ContextHandle& context,
              const SurfaceHandle& surface,
              const uint32_t min_images = 2,
              const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats =
                  {vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Srgb},
              const std::vector<vk::PresentModeKHR>& preferred_present_modes = {
                  vk::PresentModeKHR::eMailbox, vk::PresentModeKHR::eFifoRelaxed,
                  vk::PresentModeKHR::eFifo});

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
     * If the framebuffer extent is 0 or the acquire was not successful, std::nullopt is returned.
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

    // needs at least one acquire to be valid. Also nullopt after needs_recreate was
    // thrown.
    const std::optional<SwapchainInfo>& get_swapchain_info() {
        return info;
    }

    const ContextHandle& get_context() const {
        return context;
    }

    const std::vector<vk::PresentModeKHR>& get_supported_present_modes() const;

    const std::vector<vk::SurfaceFormatKHR>& get_supported_surface_formats() const;

    // Used for new Swapchains. Triggers a needs_recreate.
    // returns the actually selected one from supported_present_modes
    vk::PresentModeKHR set_new_present_mode(const vk::PresentModeKHR desired);

    // Used for new Swapchains. Triggers a needs_recreate.
    // returns the actually selected one from supported_surface_formats
    vk::SurfaceFormatKHR set_new_surface_format(const vk::SurfaceFormatKHR desired);

    // Used for new Swapchains. Triggers a needs_recreate.
    void set_min_images(const uint32_t min_images);

    const vk::PresentModeKHR& get_new_present_mode() const;

    const vk::SurfaceFormatKHR& get_new_surface_format() const;

    // ---------------------------------------------------------------------------

  private:
    /* Remember to also transition image layouts */
    vk::Extent2D create_swapchain(const uint32_t width, const uint32_t height);

    [[nodiscard]] vk::PresentModeKHR select_present_mode(const bool vsync);

  private:
    const ContextHandle context;
    const SurfaceHandle surface;

    std::vector<vk::PresentModeKHR> supported_present_modes;
    std::vector<vk::SurfaceFormatKHR> supported_surface_formats;

    vk::SurfaceFormatKHR new_surface_format;
    vk::PresentModeKHR new_present_mode;
    uint32_t new_min_images;

    vk::SwapchainKHR swapchain = VK_NULL_HANDLE;

    // ---------------------------------------------------------------------------

    std::optional<SwapchainInfo> info;

    // ---------------------------------------------------------------------------
    // See https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/api/swapchain_recreation
    // we keep here a chain of old swapchains that are cleaned up when the next acquire is
    // successful.
    std::shared_ptr<Swapchain> old_swapchain;
    uint32_t old_swapchain_chain_length = 0;

    // if > num_images => Save to destroy old swapchain
    // since it means at least one present happend.
    // We then set save_to_destoy to true for the old swapchain, and reset the pointer.
    std::size_t acquire_count = 0;

    // set by the new swapchain, if false then a deviceIdle/queueIdle is necessary when destroying.
    bool save_to_destoy = false;
    // ---------------------------------------------------------------------------

    // Contains:
    // - Semaphore (read) that is signaled by the presentation engine when the acquired image is
    // ready. access with acquire_index
    // - Semaphore (written) that must be signaled by the user when they finished writing to the
    // acquired image. access with image_idx
    // - Helper to detect if an acquire has finished on a swapchain
    //
    std::vector<SyncGroup> sync_groups;

    BinarySemaphoreHandle spare_read_semaphore;
};

} // namespace merian
