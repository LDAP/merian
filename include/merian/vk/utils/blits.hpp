#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/utils/math.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"

namespace merian {

inline void cmd_blit_stretch(const CommandBufferHandle& cmd,
                             const ImageHandle& src_image,
                             const vk::ImageLayout& src_layout,
                             const vk::Extent3D& src_extent,
                             const ImageHandle& dst_image,
                             const vk::ImageLayout& dst_layout,
                             const vk::Extent3D& dst_extent,
                             const std::optional<vk::ClearColorValue> clear_color = std::nullopt,
                             const vk::Filter filter = vk::Filter::eLinear) {
    if (clear_color) {
        cmd->clear(dst_image, dst_layout, *clear_color);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                     vk::ImageMemoryBarrier{
                         vk::AccessFlagBits::eTransferWrite,
                         vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                         dst_layout, dst_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                         *dst_image, all_levels_and_layers()});
    }

    vk::ImageBlit region{first_layer(), {}, first_layer(), {{}}};
    region.srcOffsets[1] = to_offset(src_extent);
    region.dstOffsets[1] = to_offset(dst_extent);
    cmd->blit(src_image, src_layout, dst_image, dst_layout, region, filter);
}

// Scales down and centers the src image to fit the dst image. Can lead to borders.
inline void cmd_blit_fit(const CommandBufferHandle& cmd,
                         const ImageHandle& src_image,
                         const vk::ImageLayout& src_layout,
                         const vk::Extent3D& src_extent,
                         const ImageHandle& dst_image,
                         const vk::ImageLayout& dst_layout,
                         const vk::Extent3D& dst_extent,
                         const std::optional<vk::ClearColorValue> clear_color = std::nullopt,
                         const vk::Filter filter = vk::Filter::eLinear) {
    if (clear_color) {
        cmd->clear(dst_image, dst_layout, *clear_color);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                     vk::ImageMemoryBarrier{
                         vk::AccessFlagBits::eTransferWrite,
                         vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                         dst_layout, dst_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                         *dst_image, all_levels_and_layers()});
    }

    vk::ImageBlit region{first_layer(), {}, first_layer(), {}};
    region.srcOffsets[1] = to_offset(src_extent);

    std::tie(region.dstOffsets[0], region.dstOffsets[1]) =
        fit(region.srcOffsets[0], region.srcOffsets[1], {}, to_offset(dst_extent));

    cmd->blit(src_image, src_layout, dst_image, dst_layout, region, filter);
}

// Scales up and centers the source image. Can cut of parts of the source image.
inline void cmd_blit_fill(const CommandBufferHandle& cmd,
                          const ImageHandle& src_image,
                          const vk::ImageLayout& src_layout,
                          const vk::Extent3D& src_extent,
                          const ImageHandle& dst_image,
                          const vk::ImageLayout& dst_layout,
                          const vk::Extent3D& dst_extent,
                          const std::optional<vk::ClearColorValue> clear_color = std::nullopt,
                          const vk::Filter filter = vk::Filter::eLinear) {
    if (clear_color) {
        cmd->clear(dst_image, dst_layout, *clear_color);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                     vk::ImageMemoryBarrier{
                         vk::AccessFlagBits::eTransferWrite,
                         vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                         dst_layout, dst_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                         *dst_image, all_levels_and_layers()});
    }

    vk::ImageBlit region{first_layer(), {}, first_layer(), {}};
    region.dstOffsets[1] = to_offset(dst_extent);

    std::tie(region.srcOffsets[0], region.srcOffsets[1]) =
        fit(region.dstOffsets[0], region.dstOffsets[1], {}, to_offset(src_extent));

    cmd->blit(src_image, src_layout, dst_image, dst_layout, region, filter);
}

enum BlitMode {
    FIT,
    FILL,
    STRETCH,
};

inline void cmd_blit(const BlitMode blit_mode,
                     const CommandBufferHandle& cmd,
                     const ImageHandle& src_image,
                     const vk::ImageLayout& src_layout,
                     const vk::Extent3D& src_extent,
                     const ImageHandle& dst_image,
                     const vk::ImageLayout& dst_layout,
                     const vk::Extent3D& dst_extent,
                     const std::optional<vk::ClearColorValue> clear_color = std::nullopt,
                     const vk::Filter filter = vk::Filter::eLinear) {
    switch (blit_mode) {
    case FIT:
        cmd_blit_fit(cmd, src_image, src_layout, src_extent, dst_image, dst_layout, dst_extent,
                     clear_color, filter);
        break;
    case FILL:
        cmd_blit_fill(cmd, src_image, src_layout, src_extent, dst_image, dst_layout, dst_extent,
                      clear_color, filter);
        break;
    case STRETCH:
        cmd_blit_stretch(cmd, src_image, src_layout, src_extent, dst_image, dst_layout, dst_extent,
                         clear_color, filter);
        break;
    default:
        throw std::runtime_error{"unknown blit mode"};
    }
}

} // namespace merian
