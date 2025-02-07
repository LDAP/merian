#pragma once

#include "merian/vk/context.hpp"

#include "merian/vk/memory/memory_suballocator_vma.hpp"

namespace merian {

class StagingMemoryManager : public std::enable_shared_from_this<StagingMemoryManager> {
  public:
    StagingMemoryManager(StagingMemoryManager const&) = delete;
    StagingMemoryManager& operator=(StagingMemoryManager const&) = delete;
    StagingMemoryManager() = delete;
    StagingMemoryManager(const MemoryAllocatorHandle& memory_allocator,
                         const vk::DeviceSize block_size = (vk::DeviceSize(128) * 1024 * 1024));

    ~StagingMemoryManager();

    // -------------------------------------------------------------------------

  private:
    MemoryAllocationHandle get_upload_staging_space(const vk::DeviceSize size,
                                                    BufferHandle& upload_buffer,
                                                    vk::DeviceSize& buffer_offset);

    MemoryAllocationHandle get_download_staging_space(const vk::DeviceSize size,
                                                      BufferHandle& download_buffer,
                                                      vk::DeviceSize& buffer_offset);

    // -------------------------------------------------------------------------

  public:
    /* You must make sure that "data" matches extent and format. Extent defaults to
     * image->get_extent() - offset. Size is computed from offset, extent and format by default.*/
    void cmd_to_device(const CommandBufferHandle& cmd,
                       const ImageHandle& image,
                       const void* data,
                       const vk::ImageSubresourceLayers& subresource = first_layer(),
                       const vk::Offset3D offset = {},
                       const std::optional<vk::Extent3D> optional_extent = std::nullopt);

    /* You must make sure that "data" matches extent and format. Extent defaults to
     * image->get_extent() - offset. */
    template <class T>
    void cmd_to_device(const CommandBufferHandle& cmd,
                       const ImageHandle& image,
                       const std::vector<T>& data,
                       const vk::ImageSubresourceLayers& subresource = first_layer(),
                       const vk::Offset3D offset = {},
                       const std::optional<vk::Extent3D> optional_extent = std::nullopt);

    /* Extent defaults to image->get_extent() - offset. */
    MemoryAllocationHandle
    cmd_from_device(const CommandBufferHandle& cmd,
                    const ImageHandle& image,
                    const vk::ImageSubresourceLayers& subresource = first_layer(),
                    const vk::Offset3D offset = {},
                    const std::optional<vk::Extent3D> optional_extent = std::nullopt);

    // -------------------------------------------------------------------------

    /* You must make sure that "data" matches the buffer size. Copies [data, data + size) to
     * buffer::get_buffer_address + offset. Size defaults to buffer->get_size() - offset. */
    void cmd_to_device(const CommandBufferHandle& cmd,
                       const BufferHandle& buffer,
                       const void* data,
                       const vk::DeviceSize offset = 0ul,
                       const std::optional<vk::DeviceSize> optional_size = std::nullopt);

    /* You must make sure that "data" matches the buffer size. Copies [data, data + data.get_size())
     * to buffer::get_buffer_address + offset. Size defaults to data.size() * sizeof(T).*/
    template <class T>
    void cmd_to_device(const CommandBufferHandle& cmd,
                       const BufferHandle& buffer,
                       const std::vector<T>& data,
                       const vk::DeviceSize offset = 0ul) {
        assert(offset < buffer->get_size());
        const vk::DeviceSize size = data.size() * sizeof(T);
        assert(buffer->get_size() >= size);
        assert(offset + size <= buffer->get_size());

        BufferHandle upload_buffer;
        vk::DeviceSize buffer_offset;
        const MemoryAllocationHandle memory =
            get_upload_staging_space(size, upload_buffer, buffer_offset);

        SPDLOG_DEBUG("uploading {} of data to staging buffer", format_size(size));
        memcpy(memory->map(), data, size);
        memory->unmap();

        const vk::BufferCopy copy{buffer_offset, offset, size};
        cmd->copy(upload_buffer, buffer, copy);
    }

    /* size defaults to buffer->get_size() - offset. */
    MemoryAllocationHandle
    cmd_from_device(const CommandBufferHandle& cmd,
                    const BufferHandle& buffer,
                    const vk::DeviceSize offset = 0,
                    const std::optional<vk::DeviceSize> optional_size = std::nullopt);

    // -------------------------------------------------------------------------

  private:
    const ContextHandle context;
    const MemoryAllocatorHandle allocator;
    const vk::DeviceSize block_size;
};

using StagingMemoryManagerHandle = std::shared_ptr<StagingMemoryManager>;

} // namespace merian
