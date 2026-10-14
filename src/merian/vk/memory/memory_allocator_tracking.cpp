#include "merian/vk/memory/memory_allocator_tracking.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

TrackingMemoryAllocator::TrackingMemoryAllocator(const MemoryAllocatorHandle& backing)
    : MemoryAllocator(backing->get_context()), backing(backing) {}

vk::DeviceSize TrackingMemoryAllocator::get_allocated_size() const {
    const std::lock_guard lock(mutex);
    vk::DeviceSize size = 0;
    for (const auto& [memory, allocation_size] : allocations) {
        if (!memory.expired()) {
            size += allocation_size;
        }
    }
    return size;
}

MemoryAllocationHandle
TrackingMemoryAllocator::allocate_memory(const vk::MemoryPropertyFlags required_flags,
                                         const vk::MemoryRequirements& requirements,
                                         const std::string& debug_name,
                                         const MemoryMappingType mapping_type,
                                         const vk::MemoryPropertyFlags preferred_flags,
                                         const bool dedicated,
                                         const float dedicated_priority) {
    const MemoryAllocationHandle memory =
        backing->allocate_memory(required_flags, requirements, debug_name, mapping_type,
                                 preferred_flags, dedicated, dedicated_priority);
    track(memory, requirements.size);
    return memory;
}

BufferHandle
TrackingMemoryAllocator::create_buffer(const vk::BufferCreateInfo buffer_create_info,
                                       const MemoryMappingType mapping_type,
                                       const std::string& debug_name,
                                       const std::optional<vk::DeviceSize> min_alignment) {
    const BufferHandle buffer =
        backing->create_buffer(buffer_create_info, mapping_type, debug_name, min_alignment);
    track(buffer->get_memory(), buffer->get_memory_requirements().size);
    return buffer;
}

ImageHandle TrackingMemoryAllocator::create_image(const vk::ImageCreateInfo image_create_info,
                                                  const MemoryMappingType mapping_type,
                                                  const std::string& debug_name) {
    const ImageHandle image = backing->create_image(image_create_info, mapping_type, debug_name);
    track(image->get_memory(), image->get_memory_requirements().size);
    return image;
}

std::optional<uint32_t>
TrackingMemoryAllocator::find_memory_type(const vk::BufferCreateInfo& buffer_create_info,
                                          const MemoryMappingType mapping_type) {
    return backing->find_memory_type(buffer_create_info, mapping_type);
}

std::optional<uint32_t>
TrackingMemoryAllocator::find_memory_type(const vk::ImageCreateInfo& image_create_info,
                                          const MemoryMappingType mapping_type) {
    return backing->find_memory_type(image_create_info, mapping_type);
}

void TrackingMemoryAllocator::track(const MemoryAllocationHandle& memory,
                                    const vk::DeviceSize size) {
    const std::lock_guard lock(mutex);
    constexpr size_t min_erase_size = 64;
    if (allocations.size() >= 2 * live_count_at_erase + min_erase_size) {
        erase_expired();
    }
    allocations.emplace_back(memory, size);
}

void TrackingMemoryAllocator::erase_expired() {
    std::erase_if(allocations, [](const auto& allocation) { return allocation.first.expired(); });
    live_count_at_erase = allocations.size();
}

TrackingMemoryAllocatorHandle
TrackingMemoryAllocator::create(const MemoryAllocatorHandle& backing) {
    return TrackingMemoryAllocatorHandle(new TrackingMemoryAllocator(backing));
}

} // namespace merian
