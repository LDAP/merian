#include "merian/vk/window/swapchain.hpp"

#include "merian/vk/utils/check_result.hpp"

#include <spdlog/spdlog.h>

namespace {

vk::PresentModeKHR select_best(const std::vector<vk::PresentModeKHR>& available,
                               const std::vector<vk::PresentModeKHR>& preffered) {
    assert(!available.empty());

    for (const auto& p : preffered) {
        for (const auto& a : available) {
            if (p == a) {
                return a;
            }
        }
    }

    return available[0];
}

vk::SurfaceFormatKHR select_best(const std::vector<vk::SurfaceFormatKHR>& available,
                                 const std::vector<vk::SurfaceFormatKHR>& preffered) {
    assert(!available.empty());

    for (const auto& p : preffered) {
        for (const auto& a : available) {
            if (p.format == a.format) {
                return a;
            }
        }
    }

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
                     const uint32_t min_images,
                     const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats,
                     const std::vector<vk::PresentModeKHR>& preferred_present_modes)
    : context(context), surface(surface) {
    assert(context);
    assert(surface);

    supported_surface_formats =
        context->get_physical_device()->get_physical_device().getSurfaceFormatsKHR(*surface);
    if (supported_surface_formats.empty())
        throw std::runtime_error("Surface doesn't support any surface formats!");

    supported_present_modes =
        context->get_physical_device()->get_physical_device().getSurfacePresentModesKHR(*surface);
    if (supported_surface_formats.empty())
        throw std::runtime_error("Surface doesn't support any present modes!");

    new_surface_format = select_best(supported_surface_formats, preferred_surface_formats);
    new_present_mode = select_best(supported_present_modes, preferred_present_modes);
    new_min_images = min_images;
}

Swapchain::Swapchain(const SwapchainHandle& swapchain)
    : context(swapchain->context), surface(swapchain->surface),
      supported_present_modes(swapchain->supported_present_modes),
      supported_surface_formats(swapchain->supported_surface_formats),
      new_surface_format(swapchain->new_surface_format),
      new_present_mode(swapchain->new_present_mode), new_min_images(swapchain->new_min_images),
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
        context->get_device()->get_device().waitIdle();
    }

    if (old_swapchain) {
        // if we can be destoyed every other swapchain in the chain can as well.
        old_swapchain->save_to_destoy = true;
    }

    context->get_device()->get_device().destroySwapchainKHR(swapchain);
    swapchain = VK_NULL_HANDLE;
}

// -------------------------------------------------------------------------------------

const std::vector<vk::PresentModeKHR>& Swapchain::get_supported_present_modes() const {
    return supported_present_modes;
}

const std::vector<vk::SurfaceFormatKHR>& Swapchain::get_supported_surface_formats() const {
    return supported_surface_formats;
}

// returns the actually selected one from supported_present_modes
vk::PresentModeKHR Swapchain::set_new_present_mode(const vk::PresentModeKHR desired) {
    new_present_mode = select_best(supported_present_modes, {desired, new_present_mode});

    return new_present_mode;
}

// returns the actually selected one from supported_surface_formats
vk::SurfaceFormatKHR Swapchain::set_new_surface_format(const vk::SurfaceFormatKHR desired) {
    new_surface_format = select_best(supported_surface_formats, {desired, new_surface_format});

    return new_surface_format;
}

void Swapchain::set_min_images(const uint32_t min_images) {
    new_min_images = min_images;
}

uint32_t Swapchain::get_max_image_count() {
    uint32_t surface_max_count = surface->get_capabilities().maxImageCount;
    if (surface_max_count > 0) {
        return surface_max_count;
    }
    return (uint32_t)-1;
}

const vk::PresentModeKHR& Swapchain::get_new_present_mode() const {
    return new_present_mode;
}

const vk::SurfaceFormatKHR& Swapchain::get_new_surface_format() const {
    return new_surface_format;
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

    const auto& capabilities = surface->get_capabilities();

    info = SwapchainInfo();
    info->extent = make_extent2D(capabilities, width, height);

    if (capabilities.maxImageCount > 0 && capabilities.maxImageCount < new_min_images) {
        SPDLOG_WARN("requested {} swapchain images but max is {}", new_min_images,
                    capabilities.maxImageCount);
    }

    info->min_images = std::min(std::max(capabilities.minImageCount, new_min_images) + 1,
                                capabilities.maxImageCount);

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
        new_surface_format.format,
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
        new_surface_format.colorSpace,
        info->extent,
        info->image_create_info.arrayLayers,
        info->image_create_info.usage,
        info->image_create_info.sharingMode,
        info->image_create_info.queueFamilyIndexCount,
        info->image_create_info.pQueueFamilyIndices,
        pre_transform,
        composite,
        new_present_mode,
        VK_FALSE,
        old,
    };

    swapchain = context->get_device()->get_device().createSwapchainKHR(create_info, nullptr);

    info->cur_width = width;
    info->cur_height = height;
    info->present_mode = new_present_mode;
    info->surface_format = new_surface_format;
    info->images = context->get_device()->get_device().getSwapchainImagesKHR(swapchain);

    sync_groups.resize(info->images.size());

    for (std::size_t i = 0; i < info->images.size(); i++) {
        sync_groups[i].read_semaphore = BinarySemaphore::create(context);
        sync_groups[i].written_semaphore = BinarySemaphore::create(context);
    }
    spare_read_semaphore = BinarySemaphore::create(context);

    SPDLOG_DEBUG("created swapchain ({}) {}x{} ({} {} {})",
                 fmt::ptr(static_cast<VkSwapchainKHR>(swapchain)), info->cur_width,
                 info->cur_height, vk::to_string(info->surface_format.format),
                 vk::to_string(info->surface_format.colorSpace), vk::to_string(info->present_mode));

    return info->extent;
}

std::optional<std::pair<uint32_t, Swapchain::SyncGroup>>
Swapchain::acquire(const vk::Extent2D extent, const uint64_t timeout) {

    if (extent.width == 0 || extent.height == 0) {
        SPDLOG_DEBUG("acquire failed: extent is 0");
        return std::nullopt;
    }

    if (!info.has_value() && acquire_count > 0) {
        throw needs_recreate{"previous acquire or present failed."};
    }

    assert(!swapchain || info);

    if (!swapchain) {
        create_swapchain(extent.width, extent.height);
    } else if (extent.width != info->cur_width || extent.height != info->cur_height) {
        info.reset();
        throw needs_recreate("changed framebuffer size");
    } else if (new_present_mode != info->present_mode) {
        info.reset();
        throw needs_recreate("changed present mode");
    } else if (new_surface_format != info->surface_format) {
        info.reset();
        throw needs_recreate("changed surface format");
    } else if (new_min_images > info->min_images &&
               info->min_images < surface->get_capabilities().maxImageCount) {
        info.reset();
        throw needs_recreate("changed min images");
    }

    SPDLOG_DEBUG("aquire index {}", acquire_count % sync_groups.size());

    uint32_t image_idx;
    const vk::Result result = context->get_device()->get_device().acquireNextImageKHR(
        swapchain, timeout, *spare_read_semaphore, VK_NULL_HANDLE, &image_idx);

    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        info.reset();
        throw needs_recreate(result);
    }

    if (result == vk::Result::eSuccess) {
        SPDLOG_DEBUG("aquired image index {}", image_idx);
        std::swap(spare_read_semaphore, sync_groups[image_idx].read_semaphore);
        acquire_count++;

        if (old_swapchain) {
            sync_groups[image_idx].number_acquires++;

            if (sync_groups[image_idx].number_acquires > 1) {
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

        return std::make_pair(image_idx, sync_groups[image_idx]);
    }

    if (result == vk::Result::eTimeout || result == vk::Result::eNotReady) {
        SPDLOG_DEBUG("acquire failed: {}", vk::to_string(result));
        return std::nullopt;
    }

    check_result(result, "acquire failed");
    return std::nullopt;
}

void Swapchain::present(const QueueHandle& queue, const uint32_t image_idx) {
    vk::Result result = queue->present(vk::PresentInfoKHR{
        **sync_groups[image_idx].written_semaphore,
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
