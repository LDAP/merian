#pragma once

#include "merian/vk/utils/math.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"
#include "vulkan/vulkan.hpp"

namespace merian {

inline void cmd_blit_stretch(const vk::CommandBuffer& cmd,
                             const vk::Image& src_image,
                             const vk::ImageLayout& src_layout,
                             const vk::Extent3D& src_extent,
                             const vk::Image& dst_image,
                             const vk::ImageLayout& dst_layout,
                             const vk::Extent3D& dst_extent,
                             const vk::Filter filter = vk::Filter::eLinear) {
    vk::ImageBlit region{merian::first_layer(), {}, merian::first_layer(), {{}}};
    region.srcOffsets[1] = extent_to_offset(src_extent);
    region.dstOffsets[1] = extent_to_offset(dst_extent);
    cmd.blitImage(src_image, src_layout, dst_image, dst_layout, {region}, filter);
}

// Scales down and centers the src image to fit the dst image. Can lead to borders.
inline void cmd_blit_fit(const vk::CommandBuffer& cmd,
                         const vk::Image& src_image,
                         const vk::ImageLayout& src_layout,
                         const vk::Extent3D& src_extent,
                         const vk::Image& dst_image,
                         const vk::ImageLayout& dst_layout,
                         const vk::Extent3D& dst_extent,
                         const vk::ClearColorValue clear_color = {},
                         const bool clear = true,
                         const vk::Filter filter = vk::Filter::eLinear) {
    if (clear)
        cmd.clearColorImage(dst_image, dst_layout, clear_color, {merian::all_levels_and_layers()});

    vk::ImageBlit region{merian::first_layer(), {}, merian::first_layer(), {}};
    region.srcOffsets[1] = extent_to_offset(src_extent);
    float aspect = src_extent.width / (float)src_extent.height;
    std::tie(region.dstOffsets[0], region.dstOffsets[1]) =
        center(dst_extent, clamp_aspect(aspect, dst_extent));

    cmd.blitImage(src_image, src_layout, dst_image, dst_layout, {region}, filter);
}

// Scales up and centers the source image. Can cut of parts of the source image.
inline void cmd_blit_fill(const vk::CommandBuffer& cmd,
                          const vk::Image& src_image,
                          const vk::ImageLayout& src_layout,
                          const vk::Extent3D& src_extent,
                          const vk::Image& dst_image,
                          const vk::ImageLayout& dst_layout,
                          const vk::Extent3D& dst_extent,
                          const vk::Filter filter = vk::Filter::eLinear) {
    vk::ImageBlit region{merian::first_layer(), {}, merian::first_layer(), {}};
    float aspect = dst_extent.width / (float)dst_extent.height;
    std::tie(region.srcOffsets[0], region.srcOffsets[1]) =
        center(src_extent, clamp_aspect(aspect, src_extent));
    region.dstOffsets[1] = extent_to_offset(dst_extent);

    cmd.blitImage(src_image, src_layout, dst_image, dst_layout, {region}, filter);
}

} // namespace merian
