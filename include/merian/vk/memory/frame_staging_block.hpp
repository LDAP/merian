#pragma once

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/bump_memory_allocator.hpp"
#include "merian/vk/memory/memory_allocator.hpp"

#include <optional>

namespace merian {

// Persistently-mapped, host-coherent bump-pointer upload arena. reset() at frame start; grow()
// keeps the old buffer alive on cmd so outstanding Slices stay valid.
class FrameStagingBlock {
  public:
    struct Slice {
        std::byte* ptr;
        vk::DeviceAddress addr;
        // Ref into the FrameStagingBlock — valid only until the next alloc()/grow().
        const BufferHandle& buffer;
        vk::DeviceSize buffer_offset;
    };

    FrameStagingBlock(const MemoryAllocatorHandle& allocator,
                      vk::BufferUsageFlags usage,
                      vk::DeviceSize initial_capacity,
                      std::string debug_name = "frame staging block");

    void reset();

    // Reserves `size` bytes aligned to `align`; nullopt on overflow.
    std::optional<Slice> alloc(vk::DeviceSize size, vk::DeviceSize align);

    // Caller picks the new capacity; old buffer is kept alive on cmd.
    void grow(const CommandBufferHandle& cmd, vk::DeviceSize new_capacity);

    vk::DeviceSize get_capacity() const;

  private:
    void create_block(vk::DeviceSize capacity);

    const MemoryAllocatorHandle allocator;
    const vk::BufferUsageFlags usage;
    const std::string debug_name;

    BumpMemoryAllocatorHandle block;
    vk::DeviceAddress block_device_address = 0;
};

} // namespace merian
