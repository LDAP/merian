#include "merian/vk/window/surface_swapchain.hpp"

#include "merian/vk/utils/check_result.hpp"

#include <spdlog/spdlog.h>

namespace {

vk::Extent2D make_extent2D(const vk::SurfaceCapabilitiesKHR& capabilities,
                           const uint32_t width,
                           const uint32_t height) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    return {
        std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
    };
}

uint32_t query_max_image_count(const merian::SurfaceHandle& surface) {
    const uint32_t max_image_count = surface->get_capabilities().maxImageCount;
    return max_image_count > 0 ? max_image_count : UINT32_MAX;
}

} // namespace

namespace merian {

SurfaceSwapchain::SurfaceSwapchain(
    const ContextHandle& context,
    const SurfaceHandle& surface,
    const uint32_t min_images,
    const std::vector<vk::SurfaceFormatKHR>& preferred_surface_formats,
    const std::vector<vk::PresentModeKHR>& preferred_present_modes)
    : Swapchain(
          context,
          context->get_physical_device()->get_physical_device().getSurfaceFormatsKHR(*surface),
          context->get_physical_device()->get_physical_device().getSurfacePresentModesKHR(*surface),
          min_images,
          preferred_surface_formats,
          preferred_present_modes),
      surface(surface), max_image_count(query_max_image_count(surface)) {}

SurfaceSwapchain::SurfaceSwapchain(const std::shared_ptr<SurfaceSwapchain>& swapchain)
    : Swapchain(swapchain), surface(swapchain->surface),
      max_image_count(query_max_image_count(swapchain->surface)), old_swapchain(swapchain),
      old_swapchain_chain_length(
          swapchain->old_swapchain ? (swapchain->old_swapchain_chain_length + 1) : 1) {}

SurfaceSwapchain::~SurfaceSwapchain() {
    SPDLOG_DEBUG("destroy swapchain ({})", fmt::ptr(static_cast<VkSwapchainKHR>(swapchain)));

    if (!safe_to_destroy) {
        SPDLOG_DEBUG("use device idle to ensure present operations have finished ({})",
                     fmt::ptr(this));

        context->get_device()->get_device().waitIdle();
    }

    if (old_swapchain) {
        // if this one can be destroyed, every older one in the chain can as well
        old_swapchain->safe_to_destroy = true;
    }

    context->get_device()->get_device().destroySwapchainKHR(swapchain);
    swapchain = VK_NULL_HANDLE;
}

SwapchainHandle SurfaceSwapchain::recreate() {
    return std::make_shared<SurfaceSwapchain>(
        std::static_pointer_cast<SurfaceSwapchain>(shared_from_this()));
}

uint32_t SurfaceSwapchain::get_max_image_count() {
    return max_image_count;
}

vk::ImageLayout SurfaceSwapchain::get_present_layout() const {
    return vk::ImageLayout::ePresentSrcKHR;
}

vk::Extent2D SurfaceSwapchain::create_swapchain(const uint32_t width, const uint32_t height) {
    const vk::SwapchainKHR old = old_swapchain ? old_swapchain->swapchain : vk::SwapchainKHR{};

    if (old_swapchain) {
        SPDLOG_DEBUG("recreate swapchain (old: {})",
                     fmt::ptr(static_cast<VkSwapchainKHR>(old_swapchain->swapchain)));
    } else {
        SPDLOG_DEBUG("create swapchain");
    }

    const auto& capabilities = surface->get_capabilities();
    SPDLOG_DEBUG("surface info, min images: {}, max images: {}, min extent {}x{}, max "
                 "extent: {}x{}",
                 capabilities.minImageCount, capabilities.maxImageCount,
                 capabilities.minImageExtent.width, capabilities.minImageExtent.height,
                 capabilities.maxImageExtent.width, capabilities.maxImageExtent.height);

    // A minimized window can report its restored framebuffer size while the surface has none.
    const vk::Extent2D extent = make_extent2D(capabilities, width, height);
    if (extent.width == 0 || extent.height == 0) {
        return extent;
    }

    info = SwapchainInfo();
    info->extent = extent;

    if (capabilities.maxImageCount > 0 && capabilities.maxImageCount < new_min_images) {
        SPDLOG_WARN("requested {} swapchain images but max is {}", new_min_images,
                    capabilities.maxImageCount);
    }

    info->min_images =
        std::min(std::max(capabilities.minImageCount, new_min_images) + 1, get_max_image_count());

    const vk::SurfaceTransformFlagBitsKHR pre_transform =
        capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity
            ? vk::SurfaceTransformFlagBitsKHR::eIdentity
            : capabilities.currentTransform;

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

    const vk::SwapchainCreateInfoKHR create_info{
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

    info->present_mode = new_present_mode;
    info->surface_format = new_surface_format;
    info->images = context->get_device()->get_device().getSwapchainImagesKHR(swapchain);

    sync_groups.resize(info->images.size());
    number_acquires.assign(info->images.size(), 0);

    for (std::size_t i = 0; i < info->images.size(); i++) {
        SPDLOG_TRACE("swapchain image {} of swapchain {}: {}", i,
                     fmt::ptr(static_cast<VkSwapchainKHR>(swapchain)),
                     fmt::ptr(static_cast<VkImage>(info->images[i])));
        sync_groups[i].read_semaphore = BinarySemaphore::create(context);
        sync_groups[i].written_semaphore = BinarySemaphore::create(context);
    }
    spare_read_semaphore = BinarySemaphore::create(context);

    SPDLOG_DEBUG("created swapchain ({}) {}x{} ({} {} {})",
                 fmt::ptr(static_cast<VkSwapchainKHR>(swapchain)), info->extent.width,
                 info->extent.height, vk::to_string(info->surface_format.format),
                 vk::to_string(info->surface_format.colorSpace), vk::to_string(info->present_mode));

    return info->extent;
}

std::optional<std::pair<uint32_t, Swapchain::SyncGroup>>
SurfaceSwapchain::acquire(const vk::Extent2D extent, const uint64_t timeout) {

    if (extent.width == 0 || extent.height == 0) {
        SPDLOG_DEBUG("acquire failed: extent is 0");
        return std::nullopt;
    }

    if (!info && swapchain) {
        throw needs_recreate{"previous acquire or present failed."};
    }

    if (!swapchain) {
        const vk::Extent2D surface_extent = create_swapchain(extent.width, extent.height);
        if (surface_extent.width == 0 || surface_extent.height == 0) {
            SPDLOG_DEBUG("acquire failed: surface extent is 0");
            return std::nullopt;
        }
    } else if (extent != info->extent) {
        info.reset();
        throw needs_recreate("changed framebuffer size");
    } else if (settings_changed()) {
        info.reset();
        throw needs_recreate("changed present mode or surface format");
    } else if (new_min_images > info->min_images && info->min_images < get_max_image_count()) {
        info.reset();
        throw needs_recreate("changed min images");
    } else if (suboptimal) {
        info.reset();
        throw needs_recreate(vk::Result::eSuboptimalKHR);
    }

    uint32_t image_idx;
    const vk::Result result = context->get_device()->get_device().acquireNextImageKHR(
        swapchain, timeout, *spare_read_semaphore, VK_NULL_HANDLE, &image_idx);

    if (result == vk::Result::eErrorOutOfDateKHR) {
        info.reset();
        throw needs_recreate(result);
    }

    if (result == vk::Result::eSuccess || result == vk::Result::eSuboptimalKHR) {
        suboptimal = result == vk::Result::eSuboptimalKHR;
        SPDLOG_TRACE("swapchain {} acquired image index {} ({})",
                     fmt::ptr(static_cast<VkSwapchainKHR>(swapchain)), image_idx,
                     fmt::ptr(static_cast<VkImage>(info->images[image_idx])));

        std::swap(spare_read_semaphore, sync_groups[image_idx].read_semaphore);

        if (old_swapchain) {
            number_acquires[image_idx]++;

            if (number_acquires[image_idx] > 1) {
                SPDLOG_DEBUG("present confirmed. Cleanup old swapchains.");
                old_swapchain->safe_to_destroy = true;
                old_swapchain.reset();
                old_swapchain_chain_length = 0;
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

void SurfaceSwapchain::present(const QueueHandle& queue, const uint32_t image_idx) {
    if (!info.has_value()) {
        throw needs_recreate{"previous acquire or present failed."};
    }

    SPDLOG_TRACE("swapchain {} presenting image index {} ({})",
                 fmt::ptr(static_cast<VkSwapchainKHR>(swapchain)), image_idx,
                 fmt::ptr(static_cast<VkImage>(info->images[image_idx])));

    try {
        suboptimal |= queue->present(vk::PresentInfoKHR{
                          **sync_groups[image_idx].written_semaphore,
                          swapchain,
                          image_idx,
                      }) == vk::Result::eSuboptimalKHR;
    } catch (const vk::OutOfDateKHRError&) {
        info.reset();
        throw needs_recreate(vk::Result::eErrorOutOfDateKHR);
    }
}

} // namespace merian
