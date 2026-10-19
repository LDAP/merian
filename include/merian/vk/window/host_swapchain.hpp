#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/window/host_window.hpp"
#include "merian/vk/window/swapchain.hpp"

#include <deque>

namespace merian {

// Presents into a HostWindow, as Mailbox: the window keeps the newest frame.
class HostSwapchain : public Swapchain {
  public:
    HostSwapchain(const ContextHandle& context,
                  const HostWindowHandle& window,
                  const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats = {
                      vk::Format::eR8G8B8A8Srgb});

    HostSwapchain(const std::shared_ptr<HostSwapchain>& swapchain);

    ~HostSwapchain() override;

    std::optional<std::pair<uint32_t, SyncGroup>>
    acquire(const vk::Extent2D extent, const uint64_t timeout = UINT64_MAX) override;

    void present(const QueueHandle& queue, const uint32_t image_idx) override;

    SwapchainHandle recreate() override;

    uint32_t get_max_image_count() override;

    vk::ImageLayout get_present_layout() const override;

  private:
    // A buffer neither the host nor a copy in flight uses
    BufferHandle frame_buffer();

    void create_image(const vk::Extent2D extent);

    const HostWindowHandle window;
    const QueueHandle setup_queue;
    const ResourceAllocatorHandle allocator;
    const TimelineSemaphoreHandle copied;

    ImageHandle image;
    SyncGroup sync_group;
    uint64_t presents = 0;
    std::vector<BufferHandle> frame_buffers;
    // By the copied value that completes them
    std::deque<std::pair<uint64_t, CommandBufferHandle>> copies;
};

} // namespace merian
