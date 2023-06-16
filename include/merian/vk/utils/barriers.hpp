#pragma once

#include <unordered_map>
#include <vulkan/vulkan.hpp>

namespace merian {

// Heuristic to infer access flags from image layout
inline vk::AccessFlags access_flags_for_image_layout(vk::ImageLayout layout) {
    switch (layout) {
    case vk::ImageLayout::ePreinitialized:
        return vk::AccessFlagBits::eHostWrite;
    case vk::ImageLayout::eTransferDstOptimal:
        return vk::AccessFlagBits::eTransferWrite;
    case vk::ImageLayout::eTransferSrcOptimal:
        return vk::AccessFlagBits::eTransferRead;
    case vk::ImageLayout::eColorAttachmentOptimal:
        return vk::AccessFlagBits::eColorAttachmentWrite;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        return vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        return vk::AccessFlagBits::eShaderRead;
    default:
        return vk::AccessFlags();
    }
}

// Heuristic to infer pipeline stage from image layout
inline vk::PipelineStageFlags pipeline_stage_for_image_layout(vk::ImageLayout layout) {
    switch (layout) {
    case vk::ImageLayout::eTransferDstOptimal:
    case vk::ImageLayout::eTransferSrcOptimal:
        return vk::PipelineStageFlagBits::eTransfer;
    case vk::ImageLayout::eColorAttachmentOptimal:
        return vk::PipelineStageFlagBits::eColorAttachmentOutput;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        // We do this to allow queue other than graphic
        // return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        return vk::PipelineStageFlagBits::eAllCommands;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        // We do this to allow queue other than
        // graphic return
        // VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        return vk::PipelineStageFlagBits::eAllCommands;
    case vk::ImageLayout::ePreinitialized:
        return vk::PipelineStageFlagBits::eHost;
    case vk::ImageLayout::eUndefined:
        return vk::PipelineStageFlagBits::eTopOfPipe;
    case vk::ImageLayout::ePresentSrcKHR:
        return vk::PipelineStageFlagBits::eBottomOfPipe;
    default:
        return vk::PipelineStageFlagBits::eBottomOfPipe;
    }
}

// Heuristic to infer pipeline stage from access flags
vk::PipelineStageFlags pipeline_stage_for_access_flags(vk::AccessFlags flags);

vk::ImageMemoryBarrier barrier_image_layout(vk::Image image,
                                            vk::ImageLayout old_image_layout,
                                            vk::ImageLayout new_image_layout,
                                            const vk::ImageSubresourceRange& subresource_range);

void cmd_barrier_image_layout(vk::CommandBuffer cmd,
                              vk::Image image,
                              vk::ImageLayout old_image_layout,
                              vk::ImageLayout new_image_layout,
                              const vk::ImageSubresourceRange& subresource_range);

vk::ImageMemoryBarrier barrier_image_layout(vk::Image image,
                                            vk::ImageLayout old_image_layout,
                                            vk::ImageLayout new_image_layout,
                                            vk::ImageAspectFlags aspect_mask);

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
