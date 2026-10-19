#pragma once

#include "merian/fwd.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/sync/semaphore_binary.hpp"

#include <optional>
#include <spdlog/spdlog.h>

namespace merian {

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
 * Can only be used in a loop with get_max_image_count() >= num_swapchain_images >=
 * frames_in_flight.
 */
class Swapchain : public std::enable_shared_from_this<Swapchain> {
  public:
    struct SyncGroup {
        // be aware semaphore index may not match active image index!
        BinarySemaphoreHandle read_semaphore;
        BinarySemaphoreHandle written_semaphore;
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
        vk::Extent2D extent;

        vk::PresentModeKHR present_mode;
        vk::SurfaceFormatKHR surface_format;
    };

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    virtual ~Swapchain() = default;

    // ---------------------------------------------------------------------------

    /* May throw Swapchain::needs_recreate, then replace the swapchain with recreate().
     *
     * If the framebuffer extent is 0 or the acquire was not successful, std::nullopt is returned.
     * Returns the swapchain image index and the sync group that must be used to sync access to the
     * swapchain images.
     */
    virtual std::optional<std::pair<uint32_t, SyncGroup>>
    acquire(const vk::Extent2D extent, const uint64_t timeout = UINT64_MAX) = 0;

    /* Transfers ownership of the image image_idx to the presentation engine for present.
     *
     * May throw Swapchain::needs_recreate, then replace the swapchain with recreate().
     */
    virtual void present(const QueueHandle& queue, const uint32_t image_idx) = 0;

    // Replaces this swapchain, which stays valid for its pending presents.
    virtual SwapchainHandle recreate() = 0;

    virtual uint32_t get_max_image_count() = 0;

    virtual vk::ImageLayout get_present_layout() const = 0;

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

  protected:
    Swapchain(const ContextHandle& context,
              const std::vector<vk::SurfaceFormatKHR>& supported_surface_formats,
              const std::vector<vk::PresentModeKHR>& supported_present_modes,
              const uint32_t min_images,
              const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats,
              const std::vector<vk::PresentModeKHR>& preferred_present_modes);

    Swapchain(const SwapchainHandle& previous);

    bool settings_changed() const;

  protected:
    const ContextHandle context;

    std::vector<vk::PresentModeKHR> supported_present_modes;
    std::vector<vk::SurfaceFormatKHR> supported_surface_formats;

    vk::SurfaceFormatKHR new_surface_format;
    vk::PresentModeKHR new_present_mode;
    uint32_t new_min_images;

    std::optional<SwapchainInfo> info;
};

} // namespace merian
