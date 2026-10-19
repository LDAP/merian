#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"

namespace merian {

// A tightly packed host copy of the first layer and mip level of an image, or of bytes of a
// buffer: format eUndefined, extent empty.
struct HostCopy {
    MemoryAllocationHandle memory;
    vk::Format format;
    vk::Extent3D extent;
    vk::DeviceSize size;
};

// Records the copy of src (transfer-src layout) into staging memory.
HostCopy cmd_copy_to_host(const CommandBufferHandle& cmd,
                          const StagingMemoryManagerHandle& staging,
                          const ImageHandle& src);

// As above, into dst (transfer-dst usage, host-visible memory).
HostCopy
cmd_copy_to_host(const CommandBufferHandle& cmd, const BufferHandle& dst, const ImageHandle& src);

// As above, converted to the format and extent of converted (transfer-src and -dst usage).
HostCopy cmd_copy_to_host(const CommandBufferHandle& cmd,
                          const StagingMemoryManagerHandle& staging,
                          const ImageHandle& src,
                          const ImageHandle& converted,
                          const vk::Filter filter = vk::Filter::eLinear);

HostCopy cmd_copy_to_host(const CommandBufferHandle& cmd,
                          const StagingMemoryManagerHandle& staging,
                          const BufferHandle& src,
                          const vk::DeviceSize offset,
                          const vk::DeviceSize size);

ImageHandle create_host_conversion_image(const ResourceAllocatorHandle& allocator,
                                         const vk::Format format,
                                         const vk::Extent3D& extent);

} // namespace merian
