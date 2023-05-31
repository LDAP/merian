#pragma once

#include <vulkan/vulkan.hpp>

namespace merian {

void cmdBarrierImageLayout(vk::CommandBuffer cmdbuffer, vk::Image image, vk::ImageLayout oldImageLayout,
                           vk::ImageLayout newImageLayout, const vk::ImageSubresourceRange& subresourceRange);

void cmdBarrierImageLayout(vk::CommandBuffer cmdbuffer, vk::Image image, vk::ImageLayout oldImageLayout,
                           vk::ImageLayout newImageLayout, vk::ImageAspectFlags aspectMask);

inline void cmdBarrierImageLayout(vk::CommandBuffer cmdbuffer, vk::Image image, vk::ImageLayout oldImageLayout,
                                  vk::ImageLayout newImageLayout) {
    cmdBarrierImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, vk::ImageAspectFlagBits::eColor);
}

} // namespace merian
