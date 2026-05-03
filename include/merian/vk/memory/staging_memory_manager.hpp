#pragma once

#include "merian/vk/context.hpp"

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/bump_memory_allocator.hpp"
#include "merian/vk/utils/math.hpp"

namespace merian {

class StagingMemoryManager : public std::enable_shared_from_this<StagingMemoryManager> {

  public:
    // The buffer-buffer copy from or to the staging area, depending on up or download.
    struct DeviceBufferCopy {
        BufferHandle src;
        BufferHandle dst;
        vk::BufferCopy region;
    };

    // The buffer-image copy from or to the staging area, depending on up or download.
    struct DeviceImageCopy {
        BufferHandle src;
        ImageHandle dst;
        vk::BufferImageCopy region;
    };

    // Prepared image-to-buffer download.
    // Use copy to transfer the contents into the staging area, after GPU completion, map `memory`
    // to read the data.
    struct DeviceImageDownload {
        DeviceImageCopy copy;
        MemoryAllocationHandle memory;
    };

    // Prepared image-to-buffer download.
    // Use copy to transfer the contents into the staging area, after GPU completion, map `memory`
    // to read the data.
    struct DeviceBufferDownload {
        DeviceBufferCopy copy;
        MemoryAllocationHandle memory;
    };

    // use vkCmdUpdateBuffer for sizes smaller than that
    static const vk::DeviceSize CMD_UPDATE_BUFFER_THRESHOLD = 65536;

    StagingMemoryManager(StagingMemoryManager const&) = delete;
    StagingMemoryManager& operator=(StagingMemoryManager const&) = delete;
    StagingMemoryManager() = delete;
    StagingMemoryManager(
        const MemoryAllocatorHandle& memory_allocator,
        const vk::DeviceSize block_size = vk::DeviceSize(128) * 1024 * 1024,
        const vk::BufferUsageFlags upload_usage = vk::BufferUsageFlagBits::eTransferSrc,
        const vk::BufferUsageFlags download_usage = vk::BufferUsageFlagBits::eTransferDst);

    ~StagingMemoryManager();

    // -------------------------------------------------------------------------

  public:
    // Request a staging area of size 'size' for uploads. The buffer and offset are returned to
    // queue the device copy on the command buffer.
    MemoryAllocationHandle
    get_upload_staging_space(const vk::DeviceSize size,
                             BufferHandle& upload_buffer,
                             vk::DeviceSize& buffer_offset);

    // Request a staging area of size 'size' for downloads. The buffer and offset are returned to
    // queue the device copy on the command buffer.
    MemoryAllocationHandle get_download_staging_space(const vk::DeviceSize size,
                                                      BufferHandle& download_buffer,
                                                      vk::DeviceSize& buffer_offset);

    // -------------------------------------------------------------------------

    /* Stage data and return the copy; caller records it and handles layout. */
    DeviceImageCopy to_device(const ImageHandle& image,
                              const void* data,
                              const vk::ImageSubresourceLayers& subresource = first_layer(),
                              const vk::Offset3D offset = {},
                              const std::optional<vk::Extent3D> optional_extent = std::nullopt);

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
                       const std::optional<vk::Extent3D> optional_extent = std::nullopt) {
#ifndef NDEBUG
        const vk::Extent3D extent =
            optional_extent.value_or(to_extent(image->get_extent() - offset));
        const vk::DeviceSize size = static_cast<vk::DeviceSize>(extent.width) * extent.height *
                                    extent.depth * Image::format_size(image->get_format());
        assert(data.size() * sizeof(T) >= size);
#endif

        cmd_to_device(cmd, image, data.data(), subresource, offset, optional_extent);
    }

    /* Prepare a download; caller records copy.src/copy.dst/copy.region. */
    DeviceImageDownload
    from_device(const ImageHandle& image,
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

    /* Stages the data and return the pending copy for you to record. Use cmd_to_device for a
     * vkCmdUpdateBuffer fast path on small sizes. */
    DeviceBufferCopy to_device(const BufferHandle& buffer,
                               const void* data,
                               const vk::DeviceSize offset = 0ul,
                               const std::optional<vk::DeviceSize> optional_size = std::nullopt);

    /* You must make sure that "data" matches the buffer size. Copies [data, data + size) to
     * buffer::get_buffer_address + offset. Size defaults to buffer->get_size() - offset. */
    void cmd_to_device(const CommandBufferHandle& cmd,
                       const BufferHandle& buffer,
                       const void* data,
                       const vk::DeviceSize offset = 0ul,
                       const std::optional<vk::DeviceSize> optional_size = std::nullopt);

    /* Returns a staging allocation; the caller fills it via map()/unmap(). The copy to buffer is
     * already queued in cmd. Size defaults to buffer->get_size() - offset. */
    MemoryAllocationHandle
    cmd_to_device(const CommandBufferHandle& cmd,
                  const BufferHandle& buffer,
                  const vk::DeviceSize offset = 0ul,
                  const std::optional<vk::DeviceSize> optional_size = std::nullopt);

    /* You must make sure that "data" matches the buffer size. Copies [data, data + data.get_size())
     * to buffer::get_buffer_address + offset. Size defaults to data.size() * sizeof(T).*/
    template <class T>
    void cmd_to_device(const CommandBufferHandle& cmd,
                       const BufferHandle& buffer,
                       const std::vector<T>& data,
                       const vk::DeviceSize offset = 0ul) {
        if (data.empty())
            return;
        const vk::DeviceSize size = data.size() * sizeof(T);

        cmd_to_device(cmd, buffer, data.data(), offset, size);
    }

    /* Prepare a download; caller records copy.src/copy.dst/copy.region. */
    DeviceBufferDownload
    from_device(const BufferHandle& buffer,
                const vk::DeviceSize offset = 0,
                const std::optional<vk::DeviceSize> optional_size = std::nullopt);

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
    const vk::BufferUsageFlags upload_usage;
    const vk::BufferUsageFlags download_usage;

    static constexpr vk::DeviceSize STAGING_ALIGNMENT = 16;

    BumpMemoryAllocatorHandle upload_block;
    BumpMemoryAllocatorHandle download_block;

    void create_upload_block();
    void create_download_block();

    MemoryAllocationHandle suballocate(BumpMemoryAllocatorHandle& block,
                                       vk::DeviceSize size,
                                       BufferHandle& buffer,
                                       vk::DeviceSize& buffer_offset);
};

using StagingMemoryManagerHandle = std::shared_ptr<StagingMemoryManager>;

} // namespace merian
