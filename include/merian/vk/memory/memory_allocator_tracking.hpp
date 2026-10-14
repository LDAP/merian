#pragma once

#include "merian/vk/memory/memory_allocator.hpp"

#include <mutex>

namespace merian {

class TrackingMemoryAllocator;
using TrackingMemoryAllocatorHandle = std::shared_ptr<TrackingMemoryAllocator>;

// Forwards to the backing allocator and reports how much of the memory it handed out is alive.
class TrackingMemoryAllocator : public MemoryAllocator {
  private:
    explicit TrackingMemoryAllocator(const MemoryAllocatorHandle& backing);

  public:
    vk::DeviceSize get_allocated_size() const;

    MemoryAllocationHandle
    allocate_memory(const vk::MemoryPropertyFlags required_flags,
                    const vk::MemoryRequirements& requirements,
                    const std::string& debug_name = {},
                    const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                    const vk::MemoryPropertyFlags preferred_flags = {},
                    const bool dedicated = false,
                    const float dedicated_priority = 1.0) override;

    BufferHandle
    create_buffer(const vk::BufferCreateInfo buffer_create_info,
                  const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                  const std::string& debug_name = {},
                  const std::optional<vk::DeviceSize> min_alignment = std::nullopt) override;

    ImageHandle create_image(const vk::ImageCreateInfo image_create_info,
                             const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                             const std::string& debug_name = {}) override;

    std::optional<uint32_t>
    find_memory_type(const vk::BufferCreateInfo& buffer_create_info,
                     const MemoryMappingType mapping_type = MemoryMappingType::NONE) override;

    std::optional<uint32_t>
    find_memory_type(const vk::ImageCreateInfo& image_create_info,
                     const MemoryMappingType mapping_type = MemoryMappingType::NONE) override;

  private:
    void track(const MemoryAllocationHandle& memory, const vk::DeviceSize size);

    void erase_expired();

    const MemoryAllocatorHandle backing;

    mutable std::mutex mutex;
    std::vector<std::pair<std::weak_ptr<MemoryAllocation>, vk::DeviceSize>> allocations;
    size_t live_count_at_erase = 0;

  public:
    static TrackingMemoryAllocatorHandle create(const MemoryAllocatorHandle& backing);
};

} // namespace merian
