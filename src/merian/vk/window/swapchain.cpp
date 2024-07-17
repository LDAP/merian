#include "merian/vk/window/swapchain.hpp"

#include "merian/utils/vector.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <spdlog/spdlog.h>

namespace merian {

// Helper
// ----------------------------------------------------------------

/*
 * Fifo is like "vsync on". Immediate is like "vsync off".
 * Mailbox is a hybrid between the two (gpu doesnt block if running faater than the display, but
 * screen tearing doesnt happen).
 */
[[nodiscard]] vk::PresentModeKHR Swapchain::select_present_mode(const bool vsync) {
    auto present_modes =
        context->physical_device.physical_device.getSurfacePresentModesKHR(*surface);
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

// -------------------------------------------------------------------------------------

Swapchain::Swapchain(const ContextHandle& context,
                     const SurfaceHandle& surface,
                     const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats,
                     const vk::PresentModeKHR preferred_vsync_off_mode)
    : context(context), surface(surface), preferred_surface_formats(preferred_surface_formats),
      preferred_vsync_off_mode(preferred_vsync_off_mode) {
    assert(context);
    assert(surface);

    auto surface_formats = context->physical_device.physical_device.getSurfaceFormatsKHR(*surface);
    if (surface_formats.size() == 0)
        throw std::runtime_error("Surface doesn't support any surface formats!");

    surface_format = select_surface_format(surface_formats, preferred_surface_formats);
    present_mode = select_present_mode(false);

    SPDLOG_DEBUG("selected surface format {}, color space {}, present mode {}",
                 vk::to_string(surface_format.format), vk::to_string(surface_format.colorSpace),
                 vk::to_string(present_mode));
}

Swapchain::Swapchain(const SwapchainHandle& swapchain)
    : context(swapchain->context), surface(swapchain->surface),
      preferred_surface_formats(swapchain->preferred_surface_formats),
      preferred_vsync_off_mode(swapchain->preferred_vsync_off_mode),
      surface_format(swapchain->surface_format), present_mode(swapchain->present_mode),
      old_swapchain(swapchain) {}

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

vk::Extent2D Swapchain::create_swapchain(int width, int height) {
    vk::SwapchainKHR old = VK_NULL_HANDLE;
    if (old_swapchain.expired()) {
        SPDLOG_DEBUG("create swapchain");
    } else {
        SPDLOG_DEBUG("recreate swapchain");
        old = old_swapchain.lock()->swapchain;
    }

    auto capabilities =
        context->physical_device.physical_device.getSurfaceCapabilitiesKHR(*surface);
    extent = make_extent2D(capabilities, width, height);

    min_images = capabilities.minImageCount + 1; // one extra to own
    if (capabilities.maxImageCount > 0)          // 0 means no limit
        min_images = std::min(min_images, capabilities.maxImageCount);

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
                                          min_images,
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
                                          old
                                          );

    swapchain = context->device.createSwapchainKHR(createInfo, nullptr);

    std::vector<vk::Image> swapchain_images = context->device.getSwapchainImagesKHR(swapchain);
    num_images = swapchain_images.size();
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
        semaphore_group.read_semaphore = std::make_shared<BinarySemaphore>(context);
        semaphore_group.written_semaphore = std::make_shared<BinarySemaphore>(context);

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
    cur_present_mode = present_mode;
    created = true;

    // clang-format on
    SPDLOG_DEBUG("created swapchain ({}) {}x{} ({} {} {})", fmt::ptr(this), cur_width, cur_height,
                 vk::to_string(surface_format.format), vk::to_string(surface_format.colorSpace),
                 vk::to_string(cur_present_mode));
    return extent;
}

void Swapchain::destroy_entries() {
    SPDLOG_DEBUG("destroy entries");
    for (auto& entry : entries) {
        context->device.destroyImageView(entry.imageView);
    }

    SPDLOG_DEBUG("destroy semaphores");
    semaphore_groups.resize(0);

    entries.resize(0);
    barriers.resize(0);
}

void Swapchain::destroy_swapchain() {
    SPDLOG_DEBUG("destroy swapchain ({})", fmt::ptr(this));

    for (const auto& cleanup_function : cleanup_functions) {
        cleanup_function();
    }

    if (!swapchain) {
        SPDLOG_DEBUG("swapchain already destroyed");
        return;
    }
    destroy_entries();
    context->device.destroySwapchainKHR(swapchain);
    swapchain = VK_NULL_HANDLE;
}

std::optional<SwapchainAcquireResult>
Swapchain::acquire(const std::function<vk::Extent2D()>& framebuffer_extent,
                   const uint64_t timeout) {
    const vk::Extent2D extent = framebuffer_extent();

    if (extent.width == 0 || extent.height == 0) {
        return std::nullopt;
    }

    SwapchainAcquireResult aquire_result;

    if ((extent.width != cur_width || extent.height != cur_height ||
         present_mode != cur_present_mode)) {
        if (!swapchain) {
            create_swapchain(extent.width, extent.height);
        } else {
            throw needs_recreate("changed framebuffer size");
        }
    }

    const vk::Result result = context->device.acquireNextImageKHR(
        swapchain, timeout, *current_read_semaphore(), {}, &current_image_idx);

    if (result == vk::Result::eSuccess) {
        aquire_result.image = current_image();
        aquire_result.view = current_image_view();
        aquire_result.index = current_image_index();
        aquire_result.wait_semaphore = current_read_semaphore();
        aquire_result.signal_semaphore = current_written_semaphore();
        aquire_result.extent = extent;
        aquire_result.min_images = min_images;
        aquire_result.num_images = num_images;
        aquire_result.surface_format = surface_format;
        aquire_result.did_recreate = created;
        aquire_result.old_swapchain = old_swapchain;
        created = false;

        return aquire_result;
    } else if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        throw needs_recreate(result);
    } else {
        return std::nullopt;
    }
}

void Swapchain::present(const QueueHandle& queue) {
    vk::PresentInfoKHR present_info{
        1,
        &**current_written_semaphore(), // wait until the user is done writing to the image
        1,
        &swapchain,
        &current_image_idx,
    };
    vk::Result result = queue->present(present_info);
    if (result == vk::Result::eSuccess || result == vk::Result::eSuboptimalKHR) {
        current_semaphore_idx++;
        return;
    }
    if (result == vk::Result::eErrorOutOfDateKHR) {
        // purposefully invalidate
        cur_height = 0;
        cur_width = 0;
        throw needs_recreate(result);
        return;
    }
    check_result(result, "present failed");
}

vk::ImageView Swapchain::image_view(uint32_t idx) const {
    check_size(entries, idx);
    return entries[idx].imageView;
}

vk::Image Swapchain::image(uint32_t idx) const {
    check_size(entries, idx);
    return entries[idx].image;
}

} // namespace merian
