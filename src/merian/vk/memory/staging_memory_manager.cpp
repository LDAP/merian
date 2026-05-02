#include "merian/vk/memory/staging_memory_manager.hpp"
#include <spdlog/spdlog.h>

namespace merian {

StagingMemoryManager::StagingMemoryManager(const MemoryAllocatorHandle& memory_allocator,
                                           const vk::DeviceSize block_size,
                                           const vk::BufferUsageFlags upload_usage,
                                           const vk::BufferUsageFlags download_usage)
    : context(memory_allocator->get_context()), allocator(memory_allocator), block_size(block_size),
      upload_usage(upload_usage), download_usage(download_usage) {
    create_upload_block();
}

StagingMemoryManager::~StagingMemoryManager() = default;

// -------------------------------------------------------------------------

void StagingMemoryManager::create_upload_block() {
    SPDLOG_DEBUG("creating new upload staging block ({})", format_size(block_size));
    const BufferHandle buffer = allocator->create_buffer(
        vk::BufferCreateInfo{{}, block_size, upload_usage},
        MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE, "staging upload block");
    upload_block = VMAMemorySubAllocator::create(buffer);
}

void StagingMemoryManager::create_download_block() {
    SPDLOG_DEBUG("creating new download staging block ({})", format_size(block_size));
    const BufferHandle buffer =
        allocator->create_buffer(vk::BufferCreateInfo{{}, block_size, download_usage},
                                 MemoryMappingType::HOST_ACCESS_RANDOM, "staging download block");
    download_block = VMAMemorySubAllocator::create(buffer);
}

MemoryAllocationHandle StagingMemoryManager::suballocate(VMAMemorySubAllocatorHandle& block,
                                                         const vk::DeviceSize size,
                                                         BufferHandle& buffer,
                                                         vk::DeviceSize& buffer_offset) {
    const vk::MemoryRequirements reqs{size, STAGING_ALIGNMENT, ~0u};
    const auto [virtual_alloc, offset] = block->allocate(reqs);
    buffer = block->get_base_buffer();
    buffer_offset = offset;
    return std::make_shared<VMAMemorySubAllocation>(context, block, virtual_alloc, offset, size,
                                                    true);
}

// -------------------------------------------------------------------------

MemoryAllocationHandle StagingMemoryManager::get_upload_staging_space(
    const vk::DeviceSize size, BufferHandle& upload_buffer, vk::DeviceSize& upload_buffer_offset) {

    if (size <= block_size) {
        if (upload_block->get_free_size() < size + STAGING_ALIGNMENT) {
            create_upload_block();
        }
        try {
            return suballocate(upload_block, size, upload_buffer, upload_buffer_offset);
        } catch (const AllocationFailed&) {
        }
    }

    upload_buffer = allocator->create_buffer(vk::BufferCreateInfo{{}, size, upload_usage},
                                             MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE,
                                             "staging upload (individual)");
    upload_buffer_offset = 0;
    return upload_buffer->get_memory();
}

MemoryAllocationHandle
StagingMemoryManager::get_download_staging_space(const vk::DeviceSize size,
                                                 BufferHandle& download_buffer,
                                                 vk::DeviceSize& download_buffer_offset) {

    if (size <= block_size) {
        if (!download_block || download_block->get_free_size() < size + STAGING_ALIGNMENT) {
            create_download_block();
        }
        try {
            return suballocate(download_block, size, download_buffer, download_buffer_offset);
        } catch (const AllocationFailed&) {
        }
    }

    download_buffer = allocator->create_buffer(vk::BufferCreateInfo{{}, size, download_usage},
                                               MemoryMappingType::HOST_ACCESS_RANDOM,
                                               "staging download (individual)");
    download_buffer_offset = 0;
    return download_buffer->get_memory();
}

// -------------------------------------------------------------------------

StagingMemoryManager::DeviceImageCopy
StagingMemoryManager::to_device(const ImageHandle& image,
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

    DeviceImageCopy copy;
    copy.dst = image;
    copy.region.imageSubresource = subresource;
    copy.region.imageOffset = offset;
    copy.region.imageExtent = extent;
    const MemoryAllocationHandle memory =
        get_upload_staging_space(size, copy.src, copy.region.bufferOffset);

    SPDLOG_TRACE("uploading {} of data to staging buffer", format_size(size));
    memcpy(memory->map(), data, size);
    memory->unmap();

    return copy;
}

void StagingMemoryManager::cmd_to_device(const CommandBufferHandle& cmd,
                                         const ImageHandle& image,
                                         const void* data,
                                         const vk::ImageSubresourceLayers& subresource,
                                         const vk::Offset3D offset,
                                         const std::optional<vk::Extent3D> optional_extent) {
    const DeviceImageCopy copy = to_device(image, data, subresource, offset, optional_extent);
    cmd->copy(copy.src, copy.dst, copy.region);
}

StagingMemoryManager::DeviceImageDownload
StagingMemoryManager::from_device(const ImageHandle& image,
                                  const vk::ImageSubresourceLayers& subresource,
                                  const vk::Offset3D offset,
                                  const std::optional<vk::Extent3D> optional_extent) {
    assert(offset < image->get_extent());
    assert(!optional_extent || *optional_extent <= image->get_extent());
    assert(!optional_extent || *optional_extent + offset <= image->get_extent());

    const vk::Extent3D extent = optional_extent.value_or(to_extent(image->get_extent() - offset));
    const vk::DeviceSize size = static_cast<vk::DeviceSize>(extent.width) * extent.height *
                                extent.depth * Image::format_size(image->get_format());

    DeviceImageDownload download;
    download.copy.dst = image;
    download.copy.region.imageSubresource = subresource;
    download.copy.region.imageOffset = offset;
    download.copy.region.imageExtent = extent;
    download.memory =
        get_download_staging_space(size, download.copy.src, download.copy.region.bufferOffset);

    return download;
}

MemoryAllocationHandle
StagingMemoryManager::cmd_from_device(const CommandBufferHandle& cmd,
                                      const ImageHandle& image,
                                      const vk::ImageSubresourceLayers& subresource,
                                      const vk::Offset3D offset,
                                      const std::optional<vk::Extent3D> optional_extent) {
    const DeviceImageDownload download = from_device(image, subresource, offset, optional_extent);
    cmd->copy(download.copy.dst, download.copy.src, download.copy.region);
    return download.memory;
}

// -------------------------------------------------------------------------

StagingMemoryManager::DeviceBufferCopy
StagingMemoryManager::to_device(const BufferHandle& buffer,
                                const void* data,
                                const vk::DeviceSize offset,
                                const std::optional<vk::DeviceSize> optional_size) {
    assert(offset < buffer->get_size());
    assert(!optional_size || *optional_size <= buffer->get_size());
    assert(!optional_size || *optional_size + offset <= buffer->get_size());

    const vk::DeviceSize size = optional_size.value_or(buffer->get_size() - offset);
    assert(data);

    DeviceBufferCopy copy;
    copy.dst = buffer;
    copy.region.dstOffset = offset;
    copy.region.size = size;
    const MemoryAllocationHandle memory =
        get_upload_staging_space(size, copy.src, copy.region.srcOffset);

    SPDLOG_TRACE("uploading {} of data to staging buffer", format_size(size));
    memcpy(memory->map(), data, size);
    memory->unmap();

    return copy;
}

void StagingMemoryManager::cmd_to_device(const CommandBufferHandle& cmd,
                                         const BufferHandle& buffer,
                                         const void* data,
                                         const vk::DeviceSize offset,
                                         const std::optional<vk::DeviceSize> optional_size) {

    assert(offset < buffer->get_size());
    assert(!optional_size || *optional_size <= buffer->get_size());
    assert(!optional_size || *optional_size + offset <= buffer->get_size());

    const vk::DeviceSize size = optional_size.value_or(buffer->get_size() - offset);

    if (size == 0) {
        return;
    }

    assert(data);

    if (size <= CMD_UPDATE_BUFFER_THRESHOLD && size % 4 == 0 && offset % 4 == 0) {
        cmd->update(buffer, offset, size, data);
        SPDLOG_TRACE("uploading {} of data to buffer using vkCmdUpdateBuffer", format_size(size));
    } else {
        const DeviceBufferCopy copy = to_device(buffer, data, offset, size);
        cmd->copy(copy.src, copy.dst, copy.region);
    }
}

MemoryAllocationHandle
StagingMemoryManager::cmd_to_device(const CommandBufferHandle& cmd,
                                    const BufferHandle& buffer,
                                    const vk::DeviceSize offset,
                                    const std::optional<vk::DeviceSize> optional_size) {
    assert(offset < buffer->get_size());
    assert(!optional_size || *optional_size <= buffer->get_size());
    assert(!optional_size || offset + *optional_size <= buffer->get_size());

    const vk::DeviceSize size = optional_size.value_or(buffer->get_size() - offset);

    BufferHandle upload_buffer;
    vk::DeviceSize upload_buffer_offset;
    const MemoryAllocationHandle memory =
        get_upload_staging_space(size, upload_buffer, upload_buffer_offset);

    const vk::BufferCopy copy{upload_buffer_offset, offset, size};
    cmd->copy(upload_buffer, buffer, copy);

    return memory;
}

StagingMemoryManager::DeviceBufferDownload
StagingMemoryManager::from_device(const BufferHandle& buffer,
                                  const vk::DeviceSize offset,
                                  const std::optional<vk::DeviceSize> optional_size) {
    assert(offset < buffer->get_size());
    assert(!optional_size || *optional_size <= buffer->get_size());
    assert(!optional_size || offset + *optional_size <= buffer->get_size());

    const vk::DeviceSize size = optional_size.value_or(buffer->get_size() - offset);

    DeviceBufferDownload download;
    download.copy.src = buffer;
    download.copy.region.srcOffset = offset;
    download.copy.region.size = size;
    download.memory =
        get_download_staging_space(size, download.copy.dst, download.copy.region.dstOffset);

    return download;
}

MemoryAllocationHandle
StagingMemoryManager::cmd_from_device(const CommandBufferHandle& cmd,
                                      const BufferHandle& buffer,
                                      const vk::DeviceSize offset,
                                      const std::optional<vk::DeviceSize> optional_size) {
    const DeviceBufferDownload download = from_device(buffer, offset, optional_size);
    cmd->copy(download.copy.src, download.copy.dst, download.copy.region);
    return download.memory;
}

} // namespace merian
