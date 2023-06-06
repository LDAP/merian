
#include "vulkan/vulkan.hpp"

namespace merian {

inline void cmd_copy_image(const vk::CommandBuffer& cmd,
                           const vk::Image& src,
                           const vk::ImageLayout& src_layout,
                           const vk::Image& dst,
                           const vk::ImageLayout& dst_layout,
                           const vk::Extent3D& extent) {
    vk::ImageSubresourceLayers subresource{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    vk::ImageCopy region{subresource, {}, subresource, {}, extent};
    cmd.copyImage(src, src_layout, dst, dst_layout, {region});
}

} // namespace merian
