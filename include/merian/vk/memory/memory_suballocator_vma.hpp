#pragma once

#include "merian/vk/memory/memory_allocator.hpp"

#include <cstdio>
#include <optional>
#include <spdlog/spdlog.h>
#include <vk_mem_alloc.h>

namespace merian {

class VMAMemorySubAllocator;

class VMAMemorySubAllocation : public MemoryAllocation {
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
                           VmaAllocation allocation,
                           const vk::DeviceSize offset,
                           const vk::DeviceSize size)
        : MemoryAllocation(context), allocator(allocator), offset(offset), size(size),
          m_allocation(allocation) {
        SPDLOG_TRACE("create VMA suballocation ({})", fmt::ptr(this));
    }

    // frees the memory when called
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

    ImageHandle create_aliasing_image(const vk::ImageCreateInfo& image_create_info) override;

    BufferHandle create_aliasing_buffer(const vk::BufferCreateInfo& buffer_create_info) override;

    // ------------------------------------------------------------------------------------

    MemoryAllocatorHandle get_allocator() const override;

    VmaAllocation get_allocation() const {
        return m_allocation;
    }

    void properties(Properties& props) override;

  private:
    const std::shared_ptr<VMAMemorySubAllocator> allocator;
    VmaVirtualAllocation alloc;

    const vk::DeviceSize offset;
    const vk::DeviceSize size;

    VmaAllocation m_allocation;
    mutable std::mutex allocation_mutex;
    void* mapped_memory = nullptr;
};

/**
 * @brief      A suballocator for buffers that uses the VMA algorithms.
 */
class VMAMemorySubAllocator : public MemoryAllocator {
  private:
    friend class VMAMemorySubAllocation;

  public:
    static std::shared_ptr<VMAMemorySubAllocator>
    // E.g. supply VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT if you want to use the device
    // address feature
    make_allocator(const ContextHandle& context, const VmaAllocatorCreateFlags flags = {});

  private:
    VMAMemorySubAllocator() = delete;
    explicit VMAMemorySubAllocator(const ContextHandle& context,
                                   const VmaAllocatorCreateFlags flags = {});

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

    BufferHandle
    create_buffer(const vk::BufferCreateInfo buffer_create_info,
                  const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                  const std::string& debug_name = {},
                  const std::optional<vk::DeviceSize> min_alignment = std::nullopt) override;

    ImageHandle create_image(const vk::ImageCreateInfo image_create_info,
                             const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                             const std::string& debug_name = {}) override;

    // ------------------------------------------------------------------------------------

  private:
    VmaAllocator vma_allocator;
};

} // namespace merian
