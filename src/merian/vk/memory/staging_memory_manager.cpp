#include "merian/vk/memory/staging_memory_manager.hpp"
#include <spdlog/spdlog.h>

namespace merian {

StagingMemoryManager::StagingMemoryManager(const MemoryAllocatorHandle& memory_allocator,
                                           const vk::DeviceSize block_size)
    : context(memory_allocator->get_context()), allocator(memory_allocator),
      block_size(block_size) {}

StagingMemoryManager::~StagingMemoryManager() {}

// -------------------------------------------------------------------------

MemoryAllocationHandle StagingMemoryManager::get_upload_staging_space(
    const vk::DeviceSize size, BufferHandle& upload_buffer, vk::DeviceSize& upload_buffer_offset) {

    // TODO: Use memory pool / Suballocator to prevent creating a new buffer for each upload

    upload_buffer = allocator->create_buffer(
        vk::BufferCreateInfo{{}, size, vk::BufferUsageFlagBits::eTransferSrc},
        MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE, "staging upload buffer");
    upload_buffer_offset = 0;

    return upload_buffer->get_memory();
}

MemoryAllocationHandle
StagingMemoryManager::get_download_staging_space(const vk::DeviceSize size,
                                                 BufferHandle& download_buffer,
                                                 vk::DeviceSize& download_buffer_offset) {

    // TODO: Use memory pool / Suballocator to prevent creating a new buffer for each download

    download_buffer = allocator->create_buffer(
        vk::BufferCreateInfo{{}, size, vk::BufferUsageFlagBits::eTransferDst},
        MemoryMappingType::HOST_ACCESS_RANDOM, "staging download buffer");
    download_buffer_offset = 0;

    return download_buffer->get_memory();
}

// -------------------------------------------------------------------------

void StagingMemoryManager::cmd_to_device(const CommandBufferHandle& cmd,
                                         const ImageHandle& image,
                                         const void* data,
                                         const vk::ImageSubresourceLayers& subresource,
                                         const vk::Offset3D offset,
                                         const std::optional<vk::Extent3D> optional_extent) {
    assert(data);
    assert(offset < image->get_extent());
    assert(!optional_extent || *optional_extent <= image->get_extent());
    assert(!optional_extent || *optional_extent + offset <= image->get_extent());

    const vk::Extent3D extent = optional_extent.value_or(to_extent(image->get_extent() - offset));
    const vk::DeviceSize size = static_cast<vk::DeviceSize>(extent.width) * extent.height *
                                extent.depth * Image::format_size(image->get_format());

    BufferHandle upload_buffer;
    vk::DeviceSize upload_buffer_offset;
    const MemoryAllocationHandle memory =
        get_upload_staging_space(size, upload_buffer, upload_buffer_offset);

    SPDLOG_TRACE("uploading {} of data to staging buffer", format_size(size));
    memcpy(memory->map(), data, size);
    memory->unmap();

    const vk::BufferImageCopy copy{upload_buffer_offset, 0, 0, subresource, offset, extent};
    cmd->copy(upload_buffer, image, copy);
}

MemoryAllocationHandle
StagingMemoryManager::cmd_from_device(const CommandBufferHandle& cmd,
                                      const ImageHandle& image,
                                      const vk::ImageSubresourceLayers& subresource,
                                      const vk::Offset3D offset,
                                      const std::optional<vk::Extent3D> optional_extent) {
    assert(offset < image->get_extent());
    assert(!optional_extent || *optional_extent <= image->get_extent());
    assert(!optional_extent || *optional_extent + offset <= image->get_extent());

    const vk::Extent3D extent = optional_extent.value_or(to_extent(image->get_extent() - offset));
    const vk::DeviceSize size = static_cast<vk::DeviceSize>(extent.width) * extent.height *
                                extent.depth * Image::format_size(image->get_format());

    BufferHandle download_buffer;
    vk::DeviceSize download_buffer_offset;
    const MemoryAllocationHandle memory =
        get_download_staging_space(size, download_buffer, download_buffer_offset);

    vk::BufferImageCopy copy{download_buffer_offset, 0, 0, subresource, offset, extent};
    cmd->copy(image, download_buffer, copy);

    return memory;
}

// -------------------------------------------------------------------------

void StagingMemoryManager::cmd_to_device(const CommandBufferHandle& cmd,
                                         const BufferHandle& buffer,
                                         const void* data,
                                         const vk::DeviceSize offset,
                                         const std::optional<vk::DeviceSize> optional_size) {

    assert(offset < buffer->get_size());
    assert(!optional_size || *optional_size <= buffer->get_size());
    assert(!optional_size || *optional_size + offset <= buffer->get_size());
    assert(data);

    const vk::DeviceSize size = optional_size.value_or(buffer->get_size() - offset);

    if (size <= CMD_UPDATE_BUFFER_THRESHOLD && size % 4 == 0 && offset % 4 == 0) {
        // Requirements for vkCmdUpdateBuffer are met.
        cmd->update(buffer, offset, size, data);
        SPDLOG_TRACE("uploading {} of data to buffer using vkCmdUpdateBuffer", format_size(size));
    } else {
        BufferHandle upload_buffer;
        vk::DeviceSize upload_buffer_offset;
        const MemoryAllocationHandle memory =
            get_upload_staging_space(size, upload_buffer, upload_buffer_offset);

        SPDLOG_TRACE("uploading {} of data to staging buffer", format_size(size));
        memcpy(memory->map(), data, size);
        memory->unmap();

        const vk::BufferCopy copy{upload_buffer_offset, offset, size};
        cmd->copy(upload_buffer, buffer, copy);
    }
}

MemoryAllocationHandle
StagingMemoryManager::cmd_from_device(const CommandBufferHandle& cmd,
                                      const BufferHandle& buffer,
                                      const vk::DeviceSize offset,
                                      const std::optional<vk::DeviceSize> optional_size) {
    assert(offset < buffer->get_size());
    assert(!optional_size || *optional_size < buffer->get_size());
    assert(!optional_size || offset + *optional_size < buffer->get_size());

    const vk::DeviceSize size = optional_size.value_or(buffer->get_size() - offset);

    BufferHandle download_buffer;
    vk::DeviceSize download_buffer_offset;
    const MemoryAllocationHandle memory =
        get_download_staging_space(size, download_buffer, download_buffer_offset);

    vk::BufferCopy copy{offset, download_buffer_offset, size};
    cmd->copy(buffer, download_buffer, copy);

    return memory;
}

} // namespace merian
