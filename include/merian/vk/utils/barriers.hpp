#pragma once

#include <vulkan/vulkan.hpp>

namespace merian {

// Heuristic to infer access flags from image layout
inline vk::AccessFlags access_flags_for_image_layout(const vk::ImageLayout& layout) {
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
        return {};
    }
}

// Heuristic to infer access flags from image layout
inline vk::AccessFlags2 access_flags2_for_image_layout(const vk::ImageLayout& layout) {
    switch (layout) {
    case vk::ImageLayout::ePreinitialized:
        return vk::AccessFlagBits2::eHostWrite;
    case vk::ImageLayout::eTransferDstOptimal:
        return vk::AccessFlagBits2::eTransferWrite;
    case vk::ImageLayout::eTransferSrcOptimal:
        return vk::AccessFlagBits2::eTransferRead;
    case vk::ImageLayout::eColorAttachmentOptimal:
        return vk::AccessFlagBits2::eColorAttachmentWrite;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        return vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        return vk::AccessFlagBits2::eShaderRead;
    default:
        return {};
    }
}

// Heuristic to infer pipeline stage from image layout.
// This is very conservative (i.e. attemps to include all stages that may access a layout).
// However, no extensions are taken into account. For example,
// vk::PipelineStageFlagBits::eRayTracingShaderKHR might never be included!
inline vk::PipelineStageFlags
pipeline_stage_for_image_layout(const vk::ImageLayout& layout,
                                const vk::PipelineStageFlags supported_pipeline_stages) {
    switch (layout) {
    case vk::ImageLayout::eTransferDstOptimal:
    case vk::ImageLayout::eTransferSrcOptimal:
        return vk::PipelineStageFlagBits::eTransfer;
    case vk::ImageLayout::eColorAttachmentOptimal:
        return vk::PipelineStageFlagBits::eColorAttachmentOutput;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        return supported_pipeline_stages;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        return supported_pipeline_stages;
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

// Heuristic to infer pipeline stage from image layout.
// This is very conservative (i.e. attemps to include all stages that may access a layout).
// However, no extensions are taken into account. For example,
// vk::PipelineStageFlagBits::eRayTracingShaderKHR might never be included!
inline vk::PipelineStageFlags2
pipeline_stage2_for_image_layout(const vk::ImageLayout& layout,
                                 const vk::PipelineStageFlags2 supported_pipeline_stages) {
    switch (layout) {
    case vk::ImageLayout::eTransferDstOptimal:
    case vk::ImageLayout::eTransferSrcOptimal:
        return vk::PipelineStageFlagBits2::eTransfer;
    case vk::ImageLayout::eColorAttachmentOptimal:
        return vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
        return supported_pipeline_stages;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        return supported_pipeline_stages;
    case vk::ImageLayout::ePreinitialized:
        return vk::PipelineStageFlagBits2::eHost;
    case vk::ImageLayout::eUndefined:
        return vk::PipelineStageFlagBits2::eTopOfPipe;
    case vk::ImageLayout::ePresentSrcKHR:
        return vk::PipelineStageFlagBits2::eBottomOfPipe;
    default:
        return vk::PipelineStageFlagBits2::eBottomOfPipe;
    }
}

// Heuristic to infer pipeline stage from access flags
vk::PipelineStageFlags
pipeline_stage_for_access_flags(const vk::AccessFlags& flags,
                                const vk::PipelineStageFlags supported_pipeline_stages);

// Heuristic to infer pipeline stage from access flags
vk::PipelineStageFlags2
pipeline_stage_for_access_flags2(const vk::AccessFlags2& flags,
                                 const vk::PipelineStageFlags2 supported_pipeline_stages);

// This is very conservative (i.e. attemps to include all stages that may access a layout).
// However, no extensions are taken into account. For example,
// vk::PipelineStageFlagBits::eRayTracingShaderKHR might never be included!
vk::ImageMemoryBarrier2
barrier_image_layout(const vk::Image& image,
                     const vk::ImageLayout& old_image_layout,
                     const vk::ImageLayout& new_image_layout,
                     const vk::ImageSubresourceRange& subresource_range,
                     const vk::PipelineStageFlags2 supported_pipeline_stages);

// This is very conservative (i.e. attemps to include all stages that may access a layout).
// However, no extensions are taken into account. For example,
// vk::PipelineStageFlagBits::eRayTracingShaderKHR might never be included!
vk::ImageMemoryBarrier2
barrier_image_layout(const vk::Image& image,
                     const vk::ImageLayout& old_image_layout,
                     const vk::ImageLayout& new_image_layout,
                     const vk::PipelineStageFlags2 supported_pipeline_stages,
                     const vk::ImageAspectFlags& aspect_mask = vk::ImageAspectFlagBits::eColor);

} // namespace merian
