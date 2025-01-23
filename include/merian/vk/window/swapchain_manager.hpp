#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/sync/semaphore_binary.hpp"
#include "merian/vk/window/swapchain.hpp"
#include "merian/vk/window/window.hpp"

namespace merian {

struct SwapchainAcquireResult {
    // The image and its view and index in the swap chain.
    ImageViewHandle image_view;
    uint32_t index;

    // You MUST wait on this semaphore before writing to the image. ("The
    // system" signals this semaphore when it's done presenting the
    // image and can safely be reused).
    BinarySemaphoreHandle wait_semaphore;
    // You MUST signal this semaphore when done writing to the image, and
    // before presenting it. (The system waits for this before presenting).
    BinarySemaphoreHandle signal_semaphore;

    uint16_t min_images;
    uint16_t num_images;

    // Swapchain was created or recreated.
    // You can use cmd_update_image_layouts() to update the image layouts to PresentSrc.
    bool did_recreate;
};

// Manages Swapchain recreation and SwapchainImages.
//
// This is not part of the Swapchain to prevent a cyclic dependency between SwapchainImage and
// Swapchain.
//
// Now its: SwapchainManager -> Images -> Swapchain
//                        \________________^
//
class SwapchainManager {

  public:
    SwapchainManager(const SwapchainHandle& initial_swapchain) : swapchain(initial_swapchain) {}

    // ---------------------------------------------------------------------------

    /*
     * If the framebuffer extent is 0 or the aquire was not successfull, std::nullopt is returned.
     */
    std::optional<SwapchainAcquireResult> acquire(const vk::Extent2D extent,
                                                  const uint64_t timeout = UINT64_MAX) {
        std::optional<std::pair<uint32_t, Swapchain::SyncGroup>> image_index_sync_group;

        try {
            image_index_sync_group = swapchain->acquire(extent, timeout);
        } catch (const Swapchain::needs_recreate& e) {
            return std::nullopt;
        }

        if (image_index_sync_group) {
            return make_swapchain_acquire_result(image_index_sync_group->first,
                                                 image_index_sync_group->second, false);
        }

        return std::nullopt;
    }

    /*
     * If the framebuffer extent is 0 or the aquire was not successfull, std::nullopt is returned.
     */
    std::optional<SwapchainAcquireResult>
    acquire(const std::function<vk::Extent2D()>& framebuffer_extent,
            const uint64_t timeout = UINT64_MAX,
            const uint32_t tries = 3) {

        std::optional<std::pair<uint32_t, Swapchain::SyncGroup>> image_index_sync_group;
        bool recreated = false;

        for (uint32_t t = 0; !image_index_sync_group && t < tries; t++) {
            try {
                image_index_sync_group = swapchain->acquire(framebuffer_extent(), timeout);
            } catch (const Swapchain::needs_recreate& e) {
                swapchain = std::make_shared<Swapchain>(swapchain);
                recreated = true;
            }
        }

        if (image_index_sync_group) {
            return make_swapchain_acquire_result(image_index_sync_group->first,
                                                 image_index_sync_group->second, recreated);
        }

        return std::nullopt;
    }

    /*
     * If the framebuffer extent is 0 or the aquire was not successfull, std::nullopt is returned.
     */
    std::optional<SwapchainAcquireResult> acquire(const WindowHandle& window,
                                                  const uint64_t timeout = UINT64_MAX,
                                                  const uint32_t tries = 3) {
        return acquire([&]() { return window->framebuffer_extent(); }, timeout, tries);
    }

    // ---------------------------------------------------------------------------

    const SwapchainHandle& get_swapchain() {
        return swapchain;
    }

  private:
    std::optional<SwapchainAcquireResult> make_swapchain_acquire_result(
        const uint32_t image_index, const Swapchain::SyncGroup& sync_group, bool recreated) {
        const std::optional<Swapchain::SwapchainInfo>& swapchain_info =
            swapchain->get_swapchain_info();

        assert(swapchain_info); // we call this instantly after a successful acquire.

        recreated |= image_views.empty();
        if (recreated) {
            image_views.resize(swapchain_info->images.size());

            for (uint32_t i = 0; i < swapchain_info->images.size(); i++) {

                // Image
                ImageHandle image = std::make_shared<SwapchainImage>(
                    swapchain->get_context(), swapchain_info->images[i],
                    swapchain_info->image_create_info, swapchain);

                // View
                vk::ImageViewCreateInfo create_info{
                    vk::ImageViewCreateFlagBits(),
                    *image,
                    vk::ImageViewType::e2D,
                    swapchain_info->surface_format.format,
                    {vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB,
                     vk::ComponentSwizzle::eA},
                    {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                };

                image_views[i] = ImageView::create(create_info, image);
            }
        }

        SwapchainAcquireResult aquire_result;

        aquire_result.image_view = image_views[image_index];
        aquire_result.index = image_index;
        aquire_result.wait_semaphore = sync_group.read_semaphore;
        aquire_result.signal_semaphore = sync_group.written_semaphore;
        aquire_result.min_images = swapchain_info->min_images;
        aquire_result.num_images = swapchain_info->images.size();
        aquire_result.did_recreate = recreated;

        return aquire_result;
    }

  private:
    SwapchainHandle swapchain;
    std::vector<ImageViewHandle> image_views;
};

} // namespace merian
