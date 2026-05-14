#pragma once

#include "merian/vk/memory/memory_allocator.hpp"

#include <atomic>
#include <spdlog/spdlog.h>

namespace merian {

class BumpMemoryAllocator;
using BumpMemoryAllocatorHandle = std::shared_ptr<BumpMemoryAllocator>;

// A suballocation handed out by BumpMemoryAllocator. Holds a shared reference to its allocator,
// so the underlying buffer stays alive as long as any suballocation does. Destruction is a no-op
// because bump allocators never reclaim individual slots.
class BumpMemoryAllocation : public MemoryAllocation {
  public:
    BumpMemoryAllocation() = delete;
    BumpMemoryAllocation(const BumpMemoryAllocation&) = delete;
    BumpMemoryAllocation(BumpMemoryAllocation&&) = delete;

    BumpMemoryAllocation(const ContextHandle& context,
                         const BumpMemoryAllocatorHandle& allocator,
                         const vk::DeviceSize offset,
                         const vk::DeviceSize size);

    ~BumpMemoryAllocation() override;

    void invalidate(const VkDeviceSize offset = 0,
                    const VkDeviceSize size = VK_WHOLE_SIZE) override;

    void flush(const VkDeviceSize offset = 0, const VkDeviceSize size = VK_WHOLE_SIZE) override;

    // Returns persistent mapping + offset; no atomic refcount on the hot path.
    void* map() override;

    void unmap() override;

    MemoryAllocationInfo get_memory_info() const override;

    ImageHandle create_aliasing_image(const vk::ImageCreateInfo& image_create_info,
                                      const vk::DeviceSize allocation_offset = 0ul) override;

    BufferHandle create_aliasing_buffer(const vk::BufferCreateInfo& buffer_create_info,
                                        const vk::DeviceSize allocation_offset = 0ul) override;

    void bind_to_image(const ImageHandle& image,
                       const vk::DeviceSize allocation_offset = 0ul) override;

    void bind_to_buffer(const BufferHandle& buffer,
                        const vk::DeviceSize allocation_offset = 0ul) override;

    MemoryAllocatorHandle get_allocator() const override;

    const BumpMemoryAllocatorHandle& get_bump_allocator() const;

    const vk::DeviceSize& get_size() const;

    // offset into the get_bump_allocator()->get_base_buffer()
    const vk::DeviceSize& get_offset() const;

    void properties(Properties& props) override;

  private:
    friend class BumpMemoryAllocator;

    const BumpMemoryAllocatorHandle allocator;
    const vk::DeviceSize offset;
    const vk::DeviceSize size;

    std::string name;
};

// A monotonic bump allocator over a single base buffer. Suballocations cannot be freed
// individually; the entire allocator dies when its last reference is dropped (which only happens
// after the manager replaces it AND every suballocation it handed out is released).
class BumpMemoryAllocator : public MemoryAllocator {
  private:
    explicit BumpMemoryAllocator(const BufferHandle& buffer);

  public:
    BumpMemoryAllocator() = delete;

    ~BumpMemoryAllocator() override;

    MemoryAllocationHandle
    allocate_memory(const vk::MemoryPropertyFlags required_flags,
                    const vk::MemoryRequirements& requirements,
                    const std::string& debug_name = {},
                    const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                    const vk::MemoryPropertyFlags preferred_flags = {},
                    const bool dedicated = false,
                    const float dedicated_priority = 1.0) override;

    // Low level: bumps the offset and returns the start of the new range. Throws AllocationFailed
    // when the remaining capacity is insufficient.
    vk::DeviceSize allocate(const vk::MemoryRequirements& requirements);

    // Resets the bump pointer to 0. Caller must ensure no in-flight reads of any prior
    // suballocation exist (e.g. only call after a fence/queue wait or when a parent
    // command buffer keeps the underlying buffer alive across the reuse).
    void reset();

    const BufferHandle& get_base_buffer() const;

    // Persistent host pointer to the start of the base buffer; nullptr for non-mappable buffers.
    void* get_mapped_base() const;

    vk::DeviceSize get_free_size() const;

  private:
    const BufferHandle buffer;
    const MemoryAllocationInfo buffer_info;
    vk::MemoryPropertyFlags buffer_flags;
    vk::DeviceSize buffer_alignment;

    void* mapped_base{nullptr};
    std::atomic<vk::DeviceSize> current_offset{0};

  public:
    static BumpMemoryAllocatorHandle create(const BufferHandle& buffer);
};

} // namespace merian
