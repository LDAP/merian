#pragma once

#include "merian/vk/memory/memory_allocator.hpp"

#include <cstdio>
#include <spdlog/spdlog.h>
#include <vk_mem_alloc.h>

namespace merian {

class VMAMemorySubAllocator;
using VMAMemorySubAllocatorHandle = std::shared_ptr<VMAMemorySubAllocator>;

class VMAMemorySubAllocation : public MemoryAllocation {
  private:
    friend class VMAMemorySubAllocator;

  public:
    VMAMemorySubAllocation() = delete;
    VMAMemorySubAllocation(const VMAMemorySubAllocation&) = delete;
    VMAMemorySubAllocation(VMAMemorySubAllocation&&) = delete;

    /**
     * @brief      Constructs a new instance.
     *
     */
    VMAMemorySubAllocation(const ContextHandle& context,
                           const std::shared_ptr<VMAMemorySubAllocator>& allocator,
                           VmaVirtualAllocation allocation,
                           const vk::DeviceSize offset,
                           const vk::DeviceSize size);

    ~VMAMemorySubAllocation();

    // ------------------------------------------------------------------------------------

    void invalidate(const VkDeviceSize offset = 0,
                    const VkDeviceSize size = VK_WHOLE_SIZE) override;

    void flush(const VkDeviceSize offset = 0, const VkDeviceSize size = VK_WHOLE_SIZE) override;

    // Returns a mapping to the suballocation. The offset is already accounted for.
    void* map() override;

    void unmap() override;

    // ------------------------------------------------------------------------------------

    // Retrieve detailed information about 'memHandle'
    // You should not call this to often
    MemoryAllocationInfo get_memory_info() const override;

    // ------------------------------------------------------------------------------------

    ImageHandle create_aliasing_image(const vk::ImageCreateInfo& image_create_info,
                                      const vk::DeviceSize allocation_offset = 0ul) override;

    BufferHandle create_aliasing_buffer(const vk::BufferCreateInfo& buffer_create_info,
                                        const vk::DeviceSize allocation_offset = 0ul) override;

    void bind_to_image(const ImageHandle& image,
                       const vk::DeviceSize allocation_offset = 0ul) override;

    void bind_to_buffer(const BufferHandle& buffer,
                        const vk::DeviceSize allocation_offset = 0ul) override;

    // ------------------------------------------------------------------------------------

    MemoryAllocatorHandle get_allocator() const override;

    const VMAMemorySubAllocatorHandle& get_suballocator() const;

    const vk::DeviceSize& get_size() const;

    // offset into the get_suballocator()->get_base_buffer()
    const vk::DeviceSize& get_offset() const;

    void properties(Properties& props) override;

  private:
    const std::shared_ptr<VMAMemorySubAllocator> allocator;
    VmaVirtualAllocation allocation;

    const vk::DeviceSize offset;
    const vk::DeviceSize size;

    std::string name;
};

/**
 * @brief      A suballocator for buffers that uses the VMA algorithms.
 */
class VMAMemorySubAllocator : public MemoryAllocator {
  private:
    friend class VMAMemorySubAllocation;
    friend class StagingMemoryManager;

    VMAMemorySubAllocator() = delete;
    explicit VMAMemorySubAllocator(const BufferHandle& buffer);

  public:
    ~VMAMemorySubAllocator();

    // ------------------------------------------------------------------------------------

    MemoryAllocationHandle
    allocate_memory(const vk::MemoryPropertyFlags required_flags,
                    const vk::MemoryRequirements& requirements,
                    const std::string& debug_name = {},
                    const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                    const vk::MemoryPropertyFlags preferred_flags = {},
                    const bool dedicated = false,
                    const float dedicated_priority = 1.0) override;

    // ------------------------------------------------------------------------------------

    // Returns the buffer from which this allocator allocates from.
    const BufferHandle& get_base_buffer() const;

    // Returns the VMA block for the virtual allocator.
    // You should never use this directly.
    const VmaVirtualBlock& get_vma_block() const;

  private:
    const BufferHandle buffer;
    const MemoryAllocationInfo buffer_info;
    vk::MemoryPropertyFlags buffer_flags;
    vk::DeviceSize buffer_alignment;

    VmaVirtualBlock block;

  public:
    static std::shared_ptr<VMAMemorySubAllocator> create(const BufferHandle& buffer);
};

} // namespace merian
