#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/utils/math.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"

#include <algorithm>
#include <span>

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
    vk::ImageBlit region{first_layer(), {}, first_layer(), {{}}};
    region.srcOffsets[1] = to_offset(src_extent);
    region.dstOffsets[1] = to_offset(dst_extent);

    if (clear_color && dst_extent != dst_image->get_extent()) {
        cmd->clear(dst_image, dst_layout, *clear_color);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                     vk::ImageMemoryBarrier{
                         vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferWrite,
                         dst_layout, dst_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                         *dst_image, all_levels_and_layers()});
    }

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
    vk::ImageBlit region{first_layer(), {}, first_layer(), {}};
    region.srcOffsets[1] = to_offset(src_extent);

    std::tie(region.dstOffsets[0], region.dstOffsets[1]) =
        fit(region.srcOffsets[0], region.srcOffsets[1], {}, to_offset(dst_extent));

    if (clear_color && (region.dstOffsets[0] != vk::Offset3D{} ||
                        region.dstOffsets[1] != to_offset(dst_image->get_extent()))) {
        cmd->clear(dst_image, dst_layout, *clear_color);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                     vk::ImageMemoryBarrier{
                         vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferWrite,
                         dst_layout, dst_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                         *dst_image, all_levels_and_layers()});
    }

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
    vk::ImageBlit region{first_layer(), {}, first_layer(), {}};
    region.dstOffsets[1] = to_offset(dst_extent);

    std::tie(region.srcOffsets[0], region.srcOffsets[1]) =
        fit(region.dstOffsets[0], region.dstOffsets[1], {}, to_offset(src_extent));

    if (clear_color && dst_extent != dst_image->get_extent()) {
        cmd->clear(dst_image, dst_layout, *clear_color);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                     vk::ImageMemoryBarrier{
                         vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferWrite,
                         dst_layout, dst_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                         *dst_image, all_levels_and_layers()});
    }

    cmd->blit(src_image, src_layout, dst_image, dst_layout, region, filter);
}

// Generates the mip chains by successive blits from level i-1 to i.
// Inserts barriers to eTransferDstOptimal first for images that are not already
// in that layout. Leaves all levels in eTransferSrcOptimal. The images must
// have been created with eTransferSrc and eTransferDst usage.
//
// Barriers are batched per mip level across all images, so the number of pipeline
// barriers scales with the longest mip chain instead of with the image count.
inline void cmd_generate_mipmaps(const CommandBufferHandle& cmd,
                                 const std::span<const ImageHandle> images) {
    uint32_t max_mip_levels = 0;
    for (const ImageHandle& image : images) {
        max_mip_levels = std::max(max_mip_levels, image->get_mip_levels());
    }
    if (max_mip_levels <= 1) {
        return;
    }

    std::vector<vk::ImageMemoryBarrier2> to_transfer_dst;
    for (const ImageHandle& image : images) {
        if (image->get_mip_levels() > 1 &&
            image->get_current_layout() != vk::ImageLayout::eTransferDstOptimal) {
            to_transfer_dst.emplace_back(image->barrier2(vk::ImageLayout::eTransferDstOptimal));
        }
    }
    if (!to_transfer_dst.empty()) {
        cmd->barrier(to_transfer_dst);
    }

    std::vector<vk::ImageMemoryBarrier> level_barriers;
    for (uint32_t i = 1; i <= max_mip_levels; i++) {
        // images without a mip chain lack eTransferSrc usage, they must not be transitioned
        level_barriers.clear();
        for (const ImageHandle& image : images) {
            if (image->get_mip_levels() > 1 && i <= image->get_mip_levels()) {
                level_barriers.emplace_back(
                    vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead,
                    vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, **image,
                    vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, i - 1, 1, 0, 1});
            }
        }
        if (level_barriers.empty()) {
            continue;
        }
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                     level_barriers);

        for (const ImageHandle& image : images) {
            // the last level only needs its transition to TransferSrc, no outgoing blit
            if (i >= image->get_mip_levels()) {
                continue;
            }
            const uint32_t width = image->get_extent().width;
            const uint32_t height = image->get_extent().height;
            vk::ImageBlit blit{
                vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i - 1, 0, 1},
                {},
                vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i, 0, 1},
                {}};
            // clamp: for non-square images the smaller dimension reaches 1 before the chain ends
            blit.srcOffsets[1] = vk::Offset3D{int32_t(std::max(1u, width >> (i - 1))),
                                              int32_t(std::max(1u, height >> (i - 1))), 1};
            blit.dstOffsets[1] = vk::Offset3D{int32_t(std::max(1u, width >> i)),
                                              int32_t(std::max(1u, height >> i)), 1};
            cmd->blit(image, vk::ImageLayout::eTransferSrcOptimal, image,
                      vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);
        }
    }

    for (const ImageHandle& image : images) {
        if (image->get_mip_levels() > 1) {
            image->_set_current_layout(vk::ImageLayout::eTransferSrcOptimal);
        }
    }
}

inline void cmd_generate_mipmaps(const CommandBufferHandle& cmd, const ImageHandle& image) {
    cmd_generate_mipmaps(cmd, std::span{&image, 1});
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
