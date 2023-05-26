/*
 * Some parts of this code are taken from https://github.com/nvpro-samples/nvpro_core
 * which is licensed under:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk/extension/extension_vk_glfw.hpp"
#include <spdlog/spdlog.h>
#include <vk/context.hpp>

void ExtensionVkGLFW::on_instance_created(const vk::Instance& instance) {
    auto psurf = VkSurfaceKHR(surface);
    if (glfwCreateWindowSurface(instance, window, NULL, &psurf))
        throw std::runtime_error("Surface creation failed!");
    surface = vk::SurfaceKHR(psurf);
    logger->debug("created surface");
}

void ExtensionVkGLFW::on_destroy_instance(const vk::Instance& instance) {
    logger->debug("destroy surface");
    instance.destroySurfaceKHR(surface);
}

bool ExtensionVkGLFW::accept_graphics_queue(const vk::PhysicalDevice& physical_device, std::size_t queue_family_index) {
    if (physical_device.getSurfaceSupportKHR(queue_family_index, surface)) {
        return true;
    }
    return false;
}

[[nodiscard]] vk::PresentModeKHR ExtensionVkGLFW::select_present_mode() {
    auto present_modes = physical_device.getSurfacePresentModesKHR(surface);
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
            if (present_mode == vk::PresentModeKHR::eImmediate || present_mode == vk::PresentModeKHR::eMailbox) {
                best = present_mode;
            }
        }
    }

    logger->debug("vsync disabled but mode {} could not be found! Using {}", vk::to_string(preferred_vsync_off_mode),
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

void ExtensionVkGLFW::on_physical_device_selected(const vk::PhysicalDevice& physical_device) {
    this->physical_device = physical_device;

    auto surface_formats = physical_device.getSurfaceFormatsKHR(surface);
    if (surface_formats.size() == 0)
        throw std::runtime_error("Surface doesn't support any surface formats!");

    surface_format = select_surface_format(surface_formats, preferred_surface_formats);
    present_mode = select_present_mode();

    logger->debug("selected surface format {}, color space {}", vk::to_string(surface_format.format),
                  vk::to_string(surface_format.colorSpace));
}

void ExtensionVkGLFW::on_device_created(const vk::Device& device) {
    this->device = device;
}

void ExtensionVkGLFW::on_destroy_device(const vk::Device&) {
    destroy_swapchain();
    this->device = VK_NULL_HANDLE;
    this->physical_device = VK_NULL_HANDLE;
}

vk::Extent2D make_extent2D(vk::SurfaceCapabilitiesKHR capabilities, int width, int height) {
    vk::Extent2D extent;
    if (capabilities.currentExtent.width != UINT32_MAX) {
        // If the surface size is defined, the image size must match
        extent = capabilities.currentExtent;
    } else {
        extent = vk::Extent2D{(uint32_t)width, (uint32_t)height};
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height =
            std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    return extent;
}

vk::Extent2D ExtensionVkGLFW::recreate_swapchain(int width, int height) {
    vk::SwapchainKHR old_swapchain;
    if (swapchain) {
        logger->debug("recreate swapchain");
        old_swapchain = swapchain;
    } else {
        logger->debug("create swapchain");
        old_swapchain = VK_NULL_HANDLE;
    }

    auto capabilities = physical_device.getSurfaceCapabilitiesKHR(surface);
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
        surface,
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

    swapchain = device.createSwapchainKHR(createInfo, nullptr);

    if (old_swapchain) {
        logger->debug("destroy old swapchain");
        device.destroySwapchainKHR(old_swapchain);
        destroy_entries();
    }

    std::vector<vk::Image> swapchain_images = device.getSwapchainImagesKHR(swapchain);
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
        entry.imageView = device.createImageView(createInfo);

        // Semaphore
        vk::SemaphoreCreateInfo semaphoreCreateInfo;
        semaphore_group.read_semaphore = device.createSemaphore(semaphoreCreateInfo);
        semaphore_group.written_semaphore = device.createSemaphore(semaphoreCreateInfo);

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
    logger->debug("created swapchain");
    return extent;
}

void ExtensionVkGLFW::destroy_entries() {
    logger->debug("destroy entries");
    for (auto& entry : entries) {
        device.destroyImageView(entry.imageView);
    }

    logger->debug("destroy semaphores");
    for (auto& semaphore_group : semaphore_groups) {
        device.destroySemaphore(semaphore_group.read_semaphore);
        device.destroySemaphore(semaphore_group.written_semaphore);
    }

    entries.resize(0);
    semaphore_groups.resize(0);
    barriers.resize(0);
}

void ExtensionVkGLFW::destroy_swapchain() {
    logger->debug("destroy swapchain");

    if (!swapchain) {
        logger->debug("swapchain already destroyed");
        return;
    }
    destroy_entries();
    device.destroySwapchainKHR(swapchain);
    swapchain = VK_NULL_HANDLE;
}

std::optional<ExtensionVkGLFW::SwapchainAcquireResult> ExtensionVkGLFW::aquire_custom(int width, int height) {
    ExtensionVkGLFW::SwapchainAcquireResult aquire_result;

    if (width != cur_width || height != cur_height) {
        recreate_swapchain(width, height);
        aquire_result.did_recreate = true;
    }

    for (int tries = 0; tries < 2; tries++) {
        vk::Result result =
            device.acquireNextImageKHR(swapchain, UINT64_MAX, current_read_semaphore(), {}, &current_image_idx);

        if (result == vk::Result::eSuccess) {
            aquire_result.image = current_image();
            aquire_result.view = current_image_view();
            aquire_result.index = current_image_index();
            aquire_result.wait_semaphore = current_read_semaphore();
            aquire_result.signal_semaphore = current_written_semaphore();
            aquire_result.extent = extent;

            return aquire_result;
        } else if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
            recreate_swapchain(width, height);
            continue;
        } else {
            return std::nullopt;
        }
    }

    return std::nullopt;
}

vk::Result ExtensionVkGLFW::present(vk::Queue queue) {
    vk::Semaphore& written = current_written_semaphore();
    vk::PresentInfoKHR present_info{
        1,
        &written,  // wait until the user is done writing to the image
        1,
        &swapchain,       
        &current_image_idx,                  
    };
    current_semaphore_idx++;

    return queue.presentKHR(present_info);
}
