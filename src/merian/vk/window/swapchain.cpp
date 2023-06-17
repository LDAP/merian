#include "merian/vk/window/swapchain.hpp"
#include <spdlog/spdlog.h>

namespace merian {

// Helper
// ----------------------------------------------------------------

/*
 * Fifo is like "vsync on". Immediate is like "vsync off".
 * Mailbox is a hybrid between the two (gpu doesnt block if running faater than the display, but
 * screen tearing doesnt happen).
 */
[[nodiscard]] vk::PresentModeKHR Swapchain::select_present_mode() {
    auto present_modes = context->pd_container.physical_device.getSurfacePresentModesKHR(*surface);
    if (present_modes.size() == 0)
        throw std::runtime_error("Surface doesn't support any present modes!");

    // Everyone must support FIFO
    vk::PresentModeKHR best = vk::PresentModeKHR::eFifo;

    if (vsync) {
        return best;
    } else {
        // Find a faster mode
        for (const auto& present_mode : present_modes) {
            if (present_mode == preferred_vsync_off_mode) {
                return present_mode;
            }
            if (present_mode == vk::PresentModeKHR::eImmediate ||
                present_mode == vk::PresentModeKHR::eMailbox) {
                best = present_mode;
            }
        }
    }

    SPDLOG_DEBUG("vsync disabled but mode {} could not be found! Using {}",
                 vk::to_string(preferred_vsync_off_mode), vk::to_string(best));
    return best;
}

vk::SurfaceFormatKHR select_surface_format(const std::vector<vk::SurfaceFormatKHR>& available,
                                           const std::vector<vk::SurfaceFormatKHR>& preffered) {
    if (available.empty())
        throw std::runtime_error{"no surface format available!"};

    for (const auto& preferred_format : preffered) {
        for (const auto& available_format : available) {
            if (available_format.format == preferred_format.format) {
                return available_format;
            }
        }
    }

    spdlog::warn("preferred surface format not available! using first available format!");
    return available[0];
}

void Swapchain::wait_idle() {
    if (wait_queue.has_value()) {
        wait_queue.value()->wait_idle();
    } else {
        context->device.waitIdle();
    }
}

// -------------------------------------------------------------------------------------

Swapchain::Swapchain(const SharedContext& context,
                     const SurfaceHandle& surface,
                     const std::optional<QueueHandle> wait_queue,
                     const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats,
                     const vk::PresentModeKHR preferred_vsync_off_mode)
    : context(context), surface(surface), preferred_surface_formats(preferred_surface_formats),
      preferred_vsync_off_mode(preferred_vsync_off_mode), wait_queue(wait_queue) {

    auto surface_formats = context->pd_container.physical_device.getSurfaceFormatsKHR(*surface);
    if (surface_formats.size() == 0)
        throw std::runtime_error("Surface doesn't support any surface formats!");

    surface_format = select_surface_format(surface_formats, preferred_surface_formats);
    present_mode = select_present_mode();

    SPDLOG_DEBUG("selected surface format {}, color space {}, present mode {}",
                 vk::to_string(surface_format.format), vk::to_string(surface_format.colorSpace),
                 vk::to_string(present_mode));
}

Swapchain::~Swapchain() {
    destroy_swapchain();
}

// -------------------------------------------------------------------------------------

vk::Extent2D make_extent2D(vk::SurfaceCapabilitiesKHR capabilities, int width, int height) {
    vk::Extent2D extent;
    if (capabilities.currentExtent.width != UINT32_MAX) {
        // If the surface size is defined, the image size must match
        extent = capabilities.currentExtent;
    } else {
        extent = vk::Extent2D{(uint32_t)width, (uint32_t)height};
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width,
                                  capabilities.maxImageExtent.width);
        extent.height = std::clamp(extent.height, capabilities.minImageExtent.height,
                                   capabilities.maxImageExtent.height);
    }
    return extent;
}

vk::Extent2D Swapchain::recreate_swapchain(int width, int height) {
    vk::SwapchainKHR old_swapchain;
    if (swapchain) {
        SPDLOG_DEBUG("recreate swapchain");
        old_swapchain = swapchain;
    } else {
        SPDLOG_DEBUG("create swapchain");
        old_swapchain = VK_NULL_HANDLE;
    }

    auto capabilities = context->pd_container.physical_device.getSurfaceCapabilitiesKHR(*surface);
    extent = make_extent2D(capabilities, width, height);

    uint32_t num_images = capabilities.minImageCount + 1; // one extra to own
    if (capabilities.maxImageCount > 0)                   // 0 means no limit
        num_images = std::min(num_images, capabilities.maxImageCount);

    vk::SurfaceTransformFlagBitsKHR pre_transform;
    if (capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
        pre_transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    } else {
        pre_transform = capabilities.currentTransform;
    }

    // clang-format off
    vk::SwapchainCreateInfoKHR createInfo(
                                          vk::SwapchainCreateFlagBitsKHR(),
                                          *surface,
                                          num_images,
                                          surface_format.format,
                                          surface_format.colorSpace,
                                          extent,
                                          1,
                                          vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
                                          vk::SharingMode::eExclusive,
                                          0,
                                          nullptr,
                                          pre_transform,
                                          vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                          present_mode,
                                          false,
                                          old_swapchain
                                          );

    swapchain = context->device.createSwapchainKHR(createInfo, nullptr);

    if (old_swapchain) {
        SPDLOG_DEBUG("destroy old swapchain");
        context->device.destroySwapchainKHR(old_swapchain);
        destroy_entries();
    }

    std::vector<vk::Image> swapchain_images = context->device.getSwapchainImagesKHR(swapchain);
    entries.resize(swapchain_images.size());
    semaphore_groups.resize(swapchain_images.size());
    barriers.resize(swapchain_images.size());

    for (std::size_t i = 0; i < swapchain_images.size(); i++) {
        Entry& entry = entries[i];
        SemaphoreGroup& semaphore_group = semaphore_groups[i];

        // Image
        entry.image = swapchain_images[i];

        // View
        vk::ImageViewCreateInfo createInfo(
                                           vk::ImageViewCreateFlagBits(),
                                           entry.image,
                                           vk::ImageViewType::e2D,
                                           surface_format.format,
                                           {
                                            vk::ComponentSwizzle::eR,
                                            vk::ComponentSwizzle::eG,
                                            vk::ComponentSwizzle::eB,
                                            vk::ComponentSwizzle::eA
                                        },
                                        {
                                            vk::ImageAspectFlagBits::eColor,
                                            0, 1, 0, 1
                                        }
                                        );
        entry.imageView = context->device.createImageView(createInfo);

        // Semaphore
        vk::SemaphoreCreateInfo semaphoreCreateInfo;
        semaphore_group.read_semaphore = context->device.createSemaphore(semaphoreCreateInfo);
        semaphore_group.written_semaphore = context->device.createSemaphore(semaphoreCreateInfo);

        // Barrier
        vk::ImageSubresourceRange imageSubresourceRange {
            vk::ImageAspectFlagBits::eColor,    
            0,
            VK_REMAINING_MIP_LEVELS,    
            0,
            VK_REMAINING_ARRAY_LAYERS
        };
        vk::ImageMemoryBarrier barrier{
            {},
            {},
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::ePresentSrcKHR,
            {},
            {},
            entry.image,
            imageSubresourceRange,
        };
        barriers[i] = barrier;
    }

    cur_width = width;
    cur_height = height;

    // clang-format on
    SPDLOG_DEBUG("created swapchain {}x{} ({} {} {})", cur_width, cur_height,
                 vk::to_string(surface_format.format), vk::to_string(surface_format.colorSpace),
                 vk::to_string(present_mode));
    return extent;
}

void Swapchain::destroy_entries() {
    wait_idle();

    SPDLOG_DEBUG("destroy entries");
    for (auto& entry : entries) {
        context->device.destroyImageView(entry.imageView);
    }

    SPDLOG_DEBUG("destroy semaphores");
    for (auto& semaphore_group : semaphore_groups) {
        context->device.destroySemaphore(semaphore_group.read_semaphore);
        context->device.destroySemaphore(semaphore_group.written_semaphore);
    }

    entries.resize(0);
    semaphore_groups.resize(0);
    barriers.resize(0);
}

void Swapchain::destroy_swapchain() {
    SPDLOG_DEBUG("destroy swapchain");

    if (!swapchain) {
        SPDLOG_DEBUG("swapchain already destroyed");
        return;
    }
    destroy_entries();
    context->device.destroySwapchainKHR(swapchain);
    swapchain = VK_NULL_HANDLE;
}

std::optional<SwapchainAcquireResult> Swapchain::aquire_custom(int width, int height) {
    SwapchainAcquireResult aquire_result;

    if (width != cur_width || height != cur_height) {
        recreate_swapchain(width, height);
        aquire_result.did_recreate = true;
    } else {
        aquire_result.did_recreate = false;
    }

    for (int tries = 0; tries < 2; tries++) {
        vk::Result result = context->device.acquireNextImageKHR(
            swapchain, UINT64_MAX, current_read_semaphore(), {}, &current_image_idx);

        if (result == vk::Result::eSuccess) {
            aquire_result.image = current_image();
            aquire_result.view = current_image_view();
            aquire_result.index = current_image_index();
            aquire_result.wait_semaphore = current_read_semaphore();
            aquire_result.signal_semaphore = current_written_semaphore();
            aquire_result.extent = extent;

            return aquire_result;
        } else if (result == vk::Result::eErrorOutOfDateKHR ||
                   result == vk::Result::eSuboptimalKHR) {
            recreate_swapchain(width, height);
            continue;
        } else {
            return std::nullopt;
        }
    }

    return std::nullopt;
}

vk::Result Swapchain::present(vk::Queue queue) {
    vk::Semaphore& written = current_written_semaphore();
    vk::PresentInfoKHR present_info{
        1,
        &written, // wait until the user is done writing to the image
        1,        &swapchain, &current_image_idx,
    };
    current_semaphore_idx++;

    return queue.presentKHR(present_info);
}

void Swapchain::present(Queue& queue) {
    vk::Semaphore& written = current_written_semaphore();
    vk::PresentInfoKHR present_info{
        1,
        &written, // wait until the user is done writing to the image
        1,        &swapchain, &current_image_idx,
    };
    current_semaphore_idx++;

    queue.present(present_info);
}

} // namespace merian
