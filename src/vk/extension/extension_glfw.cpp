#include "vk/extension/extension_glfw.hpp"
#include <spdlog/spdlog.h>
#include <vk/context.hpp>

void ExtensionGLFW::on_instance_created(vk::Instance& instance) {
    auto psurf = VkSurfaceKHR(surface);
    if (glfwCreateWindowSurface(instance, window, NULL, &psurf))
        throw std::runtime_error("Surface creation failed!");
    surface = vk::SurfaceKHR(psurf);
    spdlog::debug("created surface");
}

void ExtensionGLFW::on_destroy_instance(vk::Instance& instance) {
    spdlog::debug("destroy surface");
    instance.destroySurfaceKHR(surface);
}

bool ExtensionGLFW::accept_graphics_queue(vk::PhysicalDevice& physical_device, std::size_t queue_family_index) {
    if (physical_device.getSurfaceSupportKHR(queue_family_index, surface)) {
        return true;
    }
    return false;
}

void ExtensionGLFW::on_context_created(Context& context) {
    recreate_swapchain(context);
}

void ExtensionGLFW::on_destroy_context(Context& context) {
    destroy_swapchain(context);
}

vk::PresentModeKHR select_present_mode(std::vector<vk::PresentModeKHR>& present_modes) {
    for (const auto& present_mode : present_modes) {
        if (present_mode == vk::PresentModeKHR::eImmediate) {
            return present_mode;
        }
    }
    return present_modes[0];
}

vk::SurfaceFormatKHR select_surface_format(std::vector<vk::SurfaceFormatKHR>& surface_formats) {
    for (const auto& surface_format : surface_formats) {
        if (surface_format.format == vk::Format::eR8G8B8A8Srgb &&
            surface_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return surface_format;
        }
    }
    return surface_formats[0];
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

void ExtensionGLFW::recreate_swapchain(Context& context) {
    vk::SwapchainKHR old_swapchain;
    if (swapchain) {
        spdlog::debug("recreate swapchain");
        old_swapchain = swapchain;
        destroy_image_views(context);
    } else {
        spdlog::debug("create swapchain");
        old_swapchain = VK_NULL_HANDLE;
    }

    auto capabilities = context.physical_device.getSurfaceCapabilitiesKHR(surface);
    auto surface_formats = context.physical_device.getSurfaceFormatsKHR(surface);
    auto present_modes = context.physical_device.getSurfacePresentModesKHR(surface);

    if (surface_formats.size() == 0)
        throw std::runtime_error("Surface doesn't support any surface formats!");
    if (present_modes.size() == 0)
        throw std::runtime_error("Surface doesn't support any present modes!");

    // Surface format selection
    // TODO: vkdt does here different things
    surface_format = select_surface_format(surface_formats);
    // TODO: vkdt does here different things
    vk::PresentModeKHR present_mode = select_present_mode(present_modes);
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

    swapchain = context.device.createSwapchainKHR(createInfo, nullptr);
    swapchain_images = context.device.getSwapchainImagesKHR(swapchain);
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
        swapchain_image_views.push_back(context.device.createImageView(createInfo));
    }
    // clang-format on

    spdlog::debug("created swapchain");
}

void ExtensionGLFW::destroy_image_views(Context& context) {
    spdlog::debug("destroy image views");
    for (auto imageView : swapchain_image_views) {
        context.device.destroyImageView(imageView);
    }
    swapchain_image_views.resize(0);
    swapchain_images.resize(0);
}

void ExtensionGLFW::destroy_swapchain(Context& context) {
    spdlog::debug("destroy swapchain");

    if (!swapchain) {
        spdlog::debug("swapchain already destroyed");
        return;
    }
    destroy_image_views(context);
    context.device.destroySwapchainKHR(swapchain);
    swapchain = VK_NULL_HANDLE;
}
