#include "merian/vk/window/swapchain.hpp"

#include "merian/vk/utils/check_result.hpp"

#include <spdlog/spdlog.h>

namespace {

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

vk::Extent2D make_extent2D(const vk::SurfaceCapabilitiesKHR capabilities,
                           const uint32_t width,
                           const uint32_t height) {
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

} // namespace

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
    }
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

    SPDLOG_DEBUG("vsync disabled but mode {} could not be found! Using {}",
                 vk::to_string(preferred_vsync_off_mode), vk::to_string(best));
    return best;
}

// -------------------------------------------------------------------------------------

SwapchainImage::SwapchainImage(const ContextHandle& context,
                               const vk::Image& image,
                               const vk::ImageCreateInfo create_info,
                               const SwapchainHandle& swapchain)
    : Image(context,
            image,
            create_info,
            vk::ImageLayout::ePresentSrcKHR /*needed for barriers using bottom of pipe*/),
      swapchain(swapchain) {}

SwapchainImage::~SwapchainImage() {
    // prevent image destruction, swapchain (presentation engine) destroys the image
    get_image() = VK_NULL_HANDLE;
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
      present_mode(swapchain->present_mode), surface_format(swapchain->surface_format),
      old_swapchain(swapchain),
      old_swapchain_chain_length(
          swapchain->old_swapchain ? (swapchain->old_swapchain_chain_length + 1) : 1) {}

Swapchain::~Swapchain() {
    SPDLOG_DEBUG("destroy swapchain ({})", fmt::ptr(static_cast<VkSwapchainKHR>(swapchain)));

    if (!save_to_destoy) {
        SPDLOG_DEBUG("use device idle to ensure present operations have finished ({})",
                     fmt::ptr(this));

        // TODO: use
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_swapchain_maintenance1.html
        context->device.waitIdle();
    }

    if (old_swapchain) {
        // if we can be destoyed every other swapchain in the chain can as well.
        old_swapchain->save_to_destoy = true;
    }

    context->device.destroySwapchainKHR(swapchain);
    swapchain = VK_NULL_HANDLE;
}

// -------------------------------------------------------------------------------------

vk::Extent2D Swapchain::create_swapchain(const uint32_t width, const uint32_t height) {
    vk::SwapchainKHR old = VK_NULL_HANDLE;

    if (old_swapchain) {
        SPDLOG_DEBUG("recreate swapchain");
        old = old_swapchain->swapchain;
    } else {
        SPDLOG_DEBUG("create swapchain");
    }

    const auto capabilities =
        context->physical_device.physical_device.getSurfaceCapabilitiesKHR(*surface);

    info = SwapchainInfo();
    info->extent = make_extent2D(capabilities, width, height);

    info->min_images = capabilities.minImageCount + 1; // one extra to own
    if (capabilities.maxImageCount > 0)                // 0 means no limit
        info->min_images = std::min(info->min_images, capabilities.maxImageCount);

    vk::SurfaceTransformFlagBitsKHR pre_transform;
    if (capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity) {
        pre_transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    } else {
        pre_transform = capabilities.currentTransform;
    }

    // Find a supported composite type.
    vk::CompositeAlphaFlagBitsKHR composite = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    if (capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque) {
        composite = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    } else if (capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eInherit) {
        composite = vk::CompositeAlphaFlagBitsKHR::eInherit;
    } else if (capabilities.supportedCompositeAlpha &
               vk::CompositeAlphaFlagBitsKHR::ePreMultiplied) {
        composite = vk::CompositeAlphaFlagBitsKHR::ePreMultiplied;
    } else if (capabilities.supportedCompositeAlpha &
               vk::CompositeAlphaFlagBitsKHR::ePostMultiplied) {
        composite = vk::CompositeAlphaFlagBitsKHR::ePostMultiplied;
    }

    info->image_create_info = {
        {},
        vk::ImageType::e2D,
        surface_format.format,
        vk::Extent3D(info->extent, 1),
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
        vk::ImageLayout::eUndefined,
        nullptr,
    };

    vk::SwapchainCreateInfoKHR create_info{
        vk::SwapchainCreateFlagBitsKHR(),
        *surface,
        info->min_images,
        info->image_create_info.format,
        surface_format.colorSpace,
        info->extent,
        info->image_create_info.arrayLayers,
        info->image_create_info.usage,
        info->image_create_info.sharingMode,
        info->image_create_info.queueFamilyIndexCount,
        info->image_create_info.pQueueFamilyIndices,
        pre_transform,
        composite,
        present_mode,
        VK_FALSE,
        old,
    };

    swapchain = context->device.createSwapchainKHR(create_info, nullptr);

    info->cur_width = width;
    info->cur_height = height;
    info->cur_present_mode = present_mode;
    info->images = context->device.getSwapchainImagesKHR(swapchain);

    sync_groups.resize(info->images.size());
    image_idx_to_sync_group.resize(info->images.size(), -1u);

    for (std::size_t i = 0; i < info->images.size(); i++) {
        SyncGroup& sync_group = sync_groups[i];

        sync_group.read_semaphore = std::make_shared<BinarySemaphore>(context);
        sync_group.written_semaphore = std::make_shared<BinarySemaphore>(context);
        sync_group.acquire_finished = Fence::create(context);
    }

    SPDLOG_DEBUG("created swapchain ({}) {}x{} ({} {} {})",
                 fmt::ptr(static_cast<VkSwapchainKHR>(swapchain)), info->cur_width,
                 info->cur_height, vk::to_string(surface_format.format),
                 vk::to_string(surface_format.colorSpace), vk::to_string(info->cur_present_mode));

    return info->extent;
}

std::optional<std::pair<uint32_t, Swapchain::SyncGroup>>
Swapchain::acquire(const vk::Extent2D extent, const uint64_t timeout) {

    if (extent.width == 0 || extent.height == 0) {
        SPDLOG_DEBUG("acquire failed: extent is 0");
        return std::nullopt;
    }

    if (!info.has_value() && acquire_count > 0) {
        throw needs_recreate{"present failed."};
    }

    assert(!swapchain || info);

    if (!swapchain) {
        create_swapchain(extent.width, extent.height);
    } else if (extent.width != info->cur_width || extent.height != info->cur_height) {
        info.reset();
        throw needs_recreate("changed framebuffer size");
    } else if (present_mode != info->cur_present_mode) {
        info.reset();
        throw needs_recreate("changed present mode (vsync)");
    }

    const uint32_t sync_index = acquire_count % info->images.size();
    if (sync_groups[sync_index].acquire_in_progress) {
        sync_groups[sync_index].acquire_finished->wait();
        sync_groups[sync_index].acquire_finished->reset();
        sync_groups[sync_index].acquire_in_progress = false;
    }

    uint32_t image_idx;
    const vk::Result result = context->device.acquireNextImageKHR(
        swapchain, timeout, *sync_groups[sync_index].read_semaphore,
        *sync_groups[sync_index].acquire_finished, &image_idx);

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        info.reset();
        throw needs_recreate(result);
    }

    if (result == vk::Result::eSuccess) {
        acquire_count++;

        sync_groups[sync_index].acquire_in_progress = true;
        assert(image_idx_to_sync_group[image_idx] == -1u);
        image_idx_to_sync_group[image_idx] = sync_index;

        if (old_swapchain) {
            if (acquire_count > info->images.size()) {
                SPDLOG_DEBUG("present confirmed. Cleanup old swapchains.");
                old_swapchain->save_to_destoy = true;
                old_swapchain.reset();
            } else if (old_swapchain_chain_length > MAX_OLD_SWAPCHAIN_CHAIN_LENGTH) {
                SPDLOG_DEBUG("old swapchain chain lenght ({}) exceeds threshold ({}). Waiting and "
                             "clean up chain.",
                             old_swapchain_chain_length, MAX_OLD_SWAPCHAIN_CHAIN_LENGTH);
                old_swapchain.reset();
                old_swapchain_chain_length = 0;
            }
        }

        return std::make_pair(image_idx, sync_groups[sync_index]);
    }

    if (result == vk::Result::eTimeout || result == vk::Result::eNotReady) {
        SPDLOG_DEBUG("acquire failed: {}", vk::to_string(result));
        return std::nullopt;
    }

    check_result(result, "acquire failed");
    return std::nullopt;
}

void Swapchain::present(const QueueHandle& queue, const uint32_t image_idx) {
    assert(image_idx_to_sync_group[image_idx] != -1u);
    const SyncGroup& sync_group = sync_groups[image_idx_to_sync_group[image_idx]];
    image_idx_to_sync_group[image_idx] = -1u;

    vk::Result result = queue->present(vk::PresentInfoKHR{
        **sync_group.written_semaphore,
        swapchain,
        image_idx,
    });

    if (result == vk::Result::eSuccess || result == vk::Result::eSuboptimalKHR) {
        return;
    }

    if (result == vk::Result::eErrorOutOfDateKHR) {
        info.reset(); // purposefully invalidate, to signal present failed.
        throw needs_recreate(result);
        return;
    }
    check_result(result, "present failed");
}

} // namespace merian
