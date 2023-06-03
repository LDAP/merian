#pragma once

#include <vulkan/vulkan.hpp>

namespace merian {

void cmd_barrier_image_layout(vk::CommandBuffer cmd,
                              vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
                              const vk::ImageSubresourceRange& subresource_range);

void cmd_barrier_image_layout(vk::CommandBuffer cmd,
                              vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
                              vk::ImageAspectFlags aspect_mask);

inline void cmd_barrier_image_layout(vk::CommandBuffer cmd,
                                     vk::Image image,
                                     vk::ImageLayout old_image_layout,
                                     vk::ImageLayout new_image_layout) {
    cmd_barrier_image_layout(cmd, image, old_image_layout, new_image_layout,
                             vk::ImageAspectFlagBits::eColor);
}

// A barrier between compute shader write and host read
void cmd_barrier_compute_host(const vk::CommandBuffer cmd);

} // namespace merian
