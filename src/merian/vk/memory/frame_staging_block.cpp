#include "merian/vk/memory/frame_staging_block.hpp"

#include "merian/utils/string.hpp"

#include <spdlog/spdlog.h>

namespace merian {

FrameStagingBlock::FrameStagingBlock(const MemoryAllocatorHandle& allocator,
                                     const vk::BufferUsageFlags usage,
                                     const vk::DeviceSize initial_capacity,
                                     std::string debug_name)
    : allocator(allocator), usage(usage), debug_name(std::move(debug_name)) {
    create_block(initial_capacity);
}

void FrameStagingBlock::create_block(const vk::DeviceSize capacity) {
    SPDLOG_DEBUG("{}: creating block ({})", debug_name, format_size(capacity));
    const BufferHandle buffer =
        allocator->create_buffer(vk::BufferCreateInfo{{}, capacity, usage},
                                 MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE, debug_name);
    block = BumpMemoryAllocator::create(buffer);
    block_device_address =
        (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) ? buffer->get_device_address() : 0;
}

void FrameStagingBlock::reset() {
    block->reset();
}

std::optional<FrameStagingBlock::Slice> FrameStagingBlock::alloc(const vk::DeviceSize size,
                                                                 const vk::DeviceSize align) {
    const vk::MemoryRequirements reqs{size, align, ~0u};
    try {
        const vk::DeviceSize offset = block->allocate(reqs);
        return Slice{
            .ptr = static_cast<std::byte*>(block->get_mapped_base()) + offset,
            .addr = block_device_address != 0 ? block_device_address + offset : 0,
            .buffer = block->get_base_buffer(),
            .buffer_offset = offset,
        };
    } catch (const AllocationFailed&) {
        return std::nullopt;
    }
}

void FrameStagingBlock::grow(const CommandBufferHandle& cmd, const vk::DeviceSize new_capacity) {
    SPDLOG_DEBUG("FrameStagingBlock '{}' grow: {} -> {}", debug_name,
                 format_size(block->get_base_buffer()->get_size()), format_size(new_capacity));
    cmd->keep_until_pool_reset(block->get_base_buffer());
    create_block(new_capacity);
}

vk::DeviceSize FrameStagingBlock::get_capacity() const {
    return block->get_base_buffer()->get_size();
}

} // namespace merian
