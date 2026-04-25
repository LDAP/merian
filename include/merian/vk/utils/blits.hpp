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

// Generates the mip chain by successive blits from level i-1 to i.
// Inserts a barrier to eTransferDstOptimal first if the image is not already
// in that layout. Leaves all levels in eTransferSrcOptimal. The image must
// have been created with eTransferSrc and eTransferDst usage.
inline void cmd_generate_mipmaps(const CommandBufferHandle& cmd, const ImageHandle& image) {
    const uint32_t mip_levels = image->get_mip_levels();
    if (mip_levels <= 1) {
        return;
    }

    if (image->get_current_layout() != vk::ImageLayout::eTransferDstOptimal) {
        cmd->barrier(image->barrier2(vk::ImageLayout::eTransferDstOptimal));
    }

    const uint32_t width = image->get_extent().width;
    const uint32_t height = image->get_extent().height;
    for (uint32_t i = 1; i <= mip_levels; i++) {
        const vk::ImageMemoryBarrier bar{
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eTransferRead,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eTransferSrcOptimal,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            *image,
            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, i - 1, 1, 0, 1}};
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                     bar);
        // run one extra iteration to leave the last mip in TransferSrc.
        if (i == mip_levels) {
            break;
        }

        vk::ImageBlit blit{vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i - 1, 0, 1},
                           {},
                           vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i, 0, 1},
                           {}};
        blit.srcOffsets[1] = vk::Offset3D{int32_t(width >> (i - 1)), int32_t(height >> (i - 1)), 1};
        blit.dstOffsets[1] = vk::Offset3D{int32_t(width >> i), int32_t(height >> i), 1};
        cmd->blit(image, vk::ImageLayout::eTransferSrcOptimal, image,
                  vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);
    }
    image->_set_current_layout(vk::ImageLayout::eTransferSrcOptimal);
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
