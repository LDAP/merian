#include "merian/vk/utils/barriers.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

vk::ImageMemoryBarrier barrier_image_layout(vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
                              const vk::ImageSubresourceRange& subresource_range) {
    vk::ImageMemoryBarrier image_memory_barrier{
        accessFlagsForImageLayout(old_image_layout),
        accessFlagsForImageLayout(new_image_layout),
        old_image_layout,
        new_image_layout,
        // Fix for a validation issue - should be needed when vk::Image sharing mode is
        // VK_SHARING_MODE_EXCLUSIVE and the values of srcQueueFamilyIndex and dstQueueFamilyIndex
        // are equal, no ownership transfer is performed, and the barrier operates as if they were
        // both set to VK_QUEUE_FAMILY_IGNORED.
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        image,
        subresource_range,
    };
    return image_memory_barrier;
}

void cmd_barrier_image_layout(vk::CommandBuffer cmd,
                              vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
                              const vk::ImageSubresourceRange& subresource_range) {

    vk::ImageMemoryBarrier image_memory_barrier = barrier_image_layout(image, old_image_layout, new_image_layout, subresource_range);

    vk::PipelineStageFlags srcStageMask = pipelineStageForLayout(old_image_layout);
    vk::PipelineStageFlags destStageMask = pipelineStageForLayout(new_image_layout);

    cmd.pipelineBarrier(srcStageMask, destStageMask, {}, 0, nullptr, 0, nullptr, 1,
                              &image_memory_barrier);
}

vk::ImageMemoryBarrier barrier_image_layout(vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
                              vk::ImageAspectFlags aspect_mask) {
    vk::ImageSubresourceRange subresourceRange{
        aspect_mask, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS,
    };
    return barrier_image_layout(image, old_image_layout, new_image_layout, subresourceRange);
}

void cmd_barrier_image_layout(vk::CommandBuffer cmd,
                              vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
                              vk::ImageAspectFlags aspect_mask) {
    vk::ImageSubresourceRange subresourceRange{
        aspect_mask, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS,
    };

    cmd_barrier_image_layout(cmd, image, old_image_layout, new_image_layout, subresourceRange);
}

// A barrier between compute shader write and host read
void cmd_barrier_compute_host(const vk::CommandBuffer cmd) {
    vk::MemoryBarrier barrier{vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost,
                        {}, 1, &barrier, 0, nullptr, 0, nullptr);
}

} // namespace merian
