#include "merian/vk/utils/host_copy.hpp"

#include "merian/vk/utils/blits.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"

namespace merian {

HostCopy cmd_copy_to_host(const CommandBufferHandle& cmd,
                          const StagingMemoryManagerHandle& staging,
                          const ImageHandle& src) {
    const vk::Extent3D extent = src->get_extent();
    return {.memory = staging->cmd_from_device(cmd, src),
            .format = src->get_format(),
            .extent = extent,
            .size = Image::format_size(src->get_format()) * extent.width * extent.height *
                    extent.depth};
}

HostCopy
cmd_copy_to_host(const CommandBufferHandle& cmd, const BufferHandle& dst, const ImageHandle& src) {
    const vk::Extent3D extent = src->get_extent();
    cmd->copy(src, dst,
              vk::BufferImageCopy{
                  0, 0, 0, first_layer(format_aspect_flags(src->get_format())), {}, extent});
    cmd->barrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
        vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead});
    return {.memory = dst->get_memory(),
            .format = src->get_format(),
            .extent = extent,
            .size = Image::format_size(src->get_format()) * extent.width * extent.height *
                    extent.depth};
}

HostCopy cmd_copy_to_host(const CommandBufferHandle& cmd,
                          const StagingMemoryManagerHandle& staging,
                          const BufferHandle& src,
                          const vk::DeviceSize offset,
                          const vk::DeviceSize size) {
    return {.memory = staging->cmd_from_device(cmd, src, offset, size),
            .format = vk::Format::eUndefined,
            .extent = {},
            .size = size};
}

HostCopy cmd_copy_to_host(const CommandBufferHandle& cmd,
                          const StagingMemoryManagerHandle& staging,
                          const ImageHandle& src,
                          const ImageHandle& converted,
                          const vk::Filter filter) {
    cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                 converted->barrier(vk::ImageLayout::eTransferDstOptimal, {},
                                    vk::AccessFlagBits::eTransferWrite, VK_QUEUE_FAMILY_IGNORED,
                                    VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), true));
    cmd_blit_stretch(cmd, src, src->get_current_layout(), src->get_extent(), converted,
                     vk::ImageLayout::eTransferDstOptimal, converted->get_extent(), std::nullopt,
                     filter);
    cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                 converted->barrier(vk::ImageLayout::eTransferSrcOptimal,
                                    vk::AccessFlagBits::eTransferWrite,
                                    vk::AccessFlagBits::eTransferRead));
    return cmd_copy_to_host(cmd, staging, converted);
}

ImageHandle create_host_conversion_image(const ResourceAllocatorHandle& allocator,
                                         const vk::Format format,
                                         const vk::Extent3D& extent) {
    return allocator->create_image(
        vk::ImageCreateInfo{
            {},
            extent.depth > 1 ? vk::ImageType::e3D : vk::ImageType::e2D,
            format,
            extent,
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
            vk::SharingMode::eExclusive,
            {},
            {},
            vk::ImageLayout::eUndefined,
        },
        MemoryMappingType::NONE, "host copy conversion");
}

} // namespace merian
