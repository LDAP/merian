#include <vulkan/vulkan.hpp>

namespace merian {

vk::AccessFlags accessFlagsForImageLayout(vk::ImageLayout layout) {
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

vk::PipelineStageFlags pipelineStageForLayout(vk::ImageLayout layout) {
    switch (layout) {
    case vk::ImageLayout::eTransferDstOptimal:
    case vk::ImageLayout::eTransferSrcOptimal:
        return vk::PipelineStageFlagBits::eTransfer;
    case vk::ImageLayout::eColorAttachmentOptimal:
        return vk::PipelineStageFlagBits::eColorAttachmentOutput;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        return vk::PipelineStageFlagBits::eAllCommands; // We do this to allow queue other than graphic
                                                        // return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        return vk::PipelineStageFlagBits::eAllCommands; // We do this to allow queue other than graphic
                                                        // return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    case vk::ImageLayout::ePreinitialized:
        return vk::PipelineStageFlagBits::eHost;
    case vk::ImageLayout::eUndefined:
        return vk::PipelineStageFlagBits::eTopOfPipe;
    default:
        return vk::PipelineStageFlagBits::eBottomOfPipe;
    }
}

void cmdBarrierImageLayout(vk::CommandBuffer cmdbuffer, vk::Image image, vk::ImageLayout oldImageLayout,
                           vk::ImageLayout newImageLayout, const vk::ImageSubresourceRange& subresourceRange) {
    // Create an image barrier to change the layout
    vk::ImageMemoryBarrier image_memory_barrier{
        accessFlagsForImageLayout(oldImageLayout),
        accessFlagsForImageLayout(newImageLayout),
        oldImageLayout,
        newImageLayout,
        // Fix for a validation issue - should be needed when vk::Image sharing mode is VK_SHARING_MODE_EXCLUSIVE
        // and the values of srcQueueFamilyIndex and dstQueueFamilyIndex are equal, no ownership transfer is performed,
        // and the barrier operates as if they were both set to VK_QUEUE_FAMILY_IGNORED.
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        image,
        subresourceRange,
    };

    vk::PipelineStageFlags srcStageMask = pipelineStageForLayout(oldImageLayout);
    vk::PipelineStageFlags destStageMask = pipelineStageForLayout(newImageLayout);

    cmdbuffer.pipelineBarrier(srcStageMask, destStageMask, {}, 0, nullptr, 0, nullptr, 1, &image_memory_barrier);
}

void cmdBarrierImageLayout(vk::CommandBuffer cmdbuffer, vk::Image image, vk::ImageLayout oldImageLayout,
                           vk::ImageLayout newImageLayout, vk::ImageAspectFlags aspectMask) {
    vk::ImageSubresourceRange subresourceRange{
        aspectMask, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS,
    };

    cmdBarrierImageLayout(cmdbuffer, image, oldImageLayout, newImageLayout, subresourceRange);
}

} // namespace merian
