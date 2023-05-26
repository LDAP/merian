#include "vk/extension/extension_vk_glfw.hpp"
#include <spdlog/spdlog.h>
#include <vk/context.hpp>

void ExtensionVkGLFW::on_instance_created(const vk::Instance& instance) {
    auto psurf = VkSurfaceKHR(surface);
    if (glfwCreateWindowSurface(instance, window, NULL, &psurf))
        throw std::runtime_error("Surface creation failed!");
    surface = vk::SurfaceKHR(psurf);
    spdlog::debug("created surface");
}

void ExtensionVkGLFW::on_destroy_instance(const vk::Instance& instance) {
    spdlog::debug("destroy surface");
    instance.destroySurfaceKHR(surface);
}

bool ExtensionVkGLFW::accept_graphics_queue(const vk::PhysicalDevice& physical_device, std::size_t queue_family_index) {
    if (physical_device.getSurfaceSupportKHR(queue_family_index, surface)) {
        return true;
    }
    return false;
}

void ExtensionVkGLFW::on_physical_device_selected(const vk::PhysicalDevice& physical_device) {
    this->physical_device = physical_device;
}

void ExtensionVkGLFW::on_device_created(const vk::Device& device) {
    this->device = device;
    recreate_swapchain();
}

void ExtensionVkGLFW::on_destroy_device(const vk::Device&) {
    destroy_swapchain();
    this->device = VK_NULL_HANDLE;
    this->physical_device = VK_NULL_HANDLE;
}

vk::PresentModeKHR select_present_mode(std::vector<vk::PresentModeKHR>& present_modes, bool vsync,
                                       vk::PresentModeKHR preferred_vsync_off_mode) {
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
            if (present_mode == vk::PresentModeKHR::eImmediate || present_mode == vk::PresentModeKHR::eMailbox) {
                best = present_mode;
            }
        }
    }

    spdlog::debug("vsync disabled but mode {} could not be found! Using {}", vk::to_string(preferred_vsync_off_mode),
                  vk::to_string(best));
    return best;
}

vk::SurfaceFormatKHR select_surface_format(std::vector<vk::SurfaceFormatKHR>& available,
                                           std::vector<vk::SurfaceFormatKHR>& preffered) {
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

vk::Extent2D select_extent2D(vk::SurfaceCapabilitiesKHR capabilities, GLFWwindow* window) {
    vk::Extent2D extent;
    if (capabilities.currentExtent.width != UINT32_MAX) {
        extent = capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        extent = vk::Extent2D{(uint32_t)width, (uint32_t)height};
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height =
            std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    return extent;
}

void ExtensionVkGLFW::recreate_swapchain() {
    vk::SwapchainKHR old_swapchain;
    if (swapchain) {
        spdlog::debug("recreate swapchain");
        old_swapchain = swapchain;
        destroy_image_views();
    } else {
        spdlog::debug("create swapchain");
        old_swapchain = VK_NULL_HANDLE;
    }

    auto capabilities = physical_device.getSurfaceCapabilitiesKHR(surface);
    auto surface_formats = physical_device.getSurfaceFormatsKHR(surface);
    auto present_modes = physical_device.getSurfacePresentModesKHR(surface);

    if (surface_formats.size() == 0)
        throw std::runtime_error("Surface doesn't support any surface formats!");
    if (present_modes.size() == 0)
        throw std::runtime_error("Surface doesn't support any present modes!");

    surface_format = select_surface_format(surface_formats, preferred_surface_formats);
    spdlog::debug("selected surface format {}, color space {}", vk::to_string(surface_format.format),
                  vk::to_string(surface_format.colorSpace));
    // TODO: vkdt does here different things
    vk::PresentModeKHR present_mode = select_present_mode(present_modes, vsync, preferred_vsync_off_mode);
    extent2D = select_extent2D(capabilities, window);

    uint32_t num_images = capabilities.minImageCount;
    if (capabilities.maxImageCount > 0) // 0 means no limit
        num_images = std::min(num_images, capabilities.maxImageCount);

    // clang-format off
    vk::SwapchainCreateInfoKHR createInfo(
        vk::SwapchainCreateFlagBitsKHR(),
        surface,
        num_images,
        surface_format.format,
        surface_format.colorSpace,
        extent2D,
        1,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
        capabilities.currentTransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        present_mode,
        false,
        old_swapchain
    );

    swapchain = device.createSwapchainKHR(createInfo, nullptr);
    swapchain_images = device.getSwapchainImagesKHR(swapchain);
    swapchain_image_views.resize(0);
    
    for (auto image : swapchain_images) {
        vk::ImageViewCreateInfo createInfo(
            vk::ImageViewCreateFlagBits(),
            image,
            vk::ImageViewType::e2D,
            surface_format.format,
            {
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity
            },
            {
                vk::ImageAspectFlagBits::eColor,
                0, 1, 0, 1
            }
        );
        swapchain_image_views.push_back(device.createImageView(createInfo));
    }
    // clang-format on
    spdlog::debug("created swapchain");

    if (old_swapchain) {
        spdlog::debug("destroy old swapchain");
        device.destroySwapchainKHR(old_swapchain);
    }
}

void ExtensionVkGLFW::destroy_image_views() {
    spdlog::debug("destroy image views");
    for (auto imageView : swapchain_image_views) {
        device.destroyImageView(imageView);
    }
    swapchain_image_views.resize(0);
    swapchain_images.resize(0);
}

void ExtensionVkGLFW::destroy_swapchain() {
    spdlog::debug("destroy swapchain");

    if (!swapchain) {
        spdlog::debug("swapchain already destroyed");
        return;
    }
    destroy_image_views();
    device.destroySwapchainKHR(swapchain);
    swapchain = VK_NULL_HANDLE;
}
