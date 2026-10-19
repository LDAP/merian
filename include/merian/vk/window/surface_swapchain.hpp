#pragma once

#include "merian/vk/window/surface.hpp"
#include "merian/vk/window/swapchain.hpp"

namespace merian {

class SurfaceSwapchain : public Swapchain {
    static constexpr uint32_t MAX_OLD_SWAPCHAIN_CHAIN_LENGTH = 5;

  public:
    /**
     * @param[in]  preferred_surface_formats  The preferred surface formats in decreasing priority;
     * without a supported one, the first supported format
     */
    SurfaceSwapchain(const ContextHandle& context,
                     const SurfaceHandle& surface,
                     const uint32_t min_images = 2,
                     const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats =
                         {vk::Format::eR8G8B8A8Srgb, vk::Format::eB8G8R8A8Srgb},
                     const std::vector<vk::PresentModeKHR>& preferred_present_modes = {
                         vk::PresentModeKHR::eMailbox, vk::PresentModeKHR::eFifoRelaxed,
                         vk::PresentModeKHR::eFifo});

    // Recreates the swapchain; the old one stays alive until its presents finished.
    SurfaceSwapchain(const std::shared_ptr<SurfaceSwapchain>& swapchain);

    ~SurfaceSwapchain() override;

    std::optional<std::pair<uint32_t, SyncGroup>>
    acquire(const vk::Extent2D extent, const uint64_t timeout = UINT64_MAX) override;

    void present(const QueueHandle& queue, const uint32_t image_idx) override;

    SwapchainHandle recreate() override;

    uint32_t get_max_image_count() override;

    vk::ImageLayout get_present_layout() const override;

  private:
    vk::Extent2D create_swapchain(const uint32_t width, const uint32_t height);

  private:
    const SurfaceHandle surface;
    const uint32_t max_image_count;

    vk::SwapchainKHR swapchain = VK_NULL_HANDLE;

    // A chain of old swapchains, released once an acquire succeeded, see
    // https://github.com/KhronosGroup/Vulkan-Samples/tree/main/samples/api/swapchain_recreation
    std::shared_ptr<SurfaceSwapchain> old_swapchain;
    uint32_t old_swapchain_chain_length = 0;

    // set by the newer swapchain; else destruction waits for the device to idle
    bool safe_to_destroy = false;
    bool suboptimal = false;

    std::vector<SyncGroup> sync_groups;
    // per image index
    std::vector<uint64_t> number_acquires;

    BinarySemaphoreHandle spare_read_semaphore;
};

} // namespace merian
