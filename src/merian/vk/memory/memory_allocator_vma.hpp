#pragma once

#include "merian/vk/memory/memory_allocator.hpp"

#include <cstdio>
#include <optional>
#include <spdlog/spdlog.h>
#include <vk_mem_alloc.h>

namespace merian {

class VMAMemoryAllocator;

class VMAMemoryAllocation : public MemoryAllocation {
  public:
    VMAMemoryAllocation() = delete;
    VMAMemoryAllocation(const VMAMemoryAllocation&) = delete;
    VMAMemoryAllocation(VMAMemoryAllocation&&) = delete;

    /**
     * @brief      Constructs a new instance.
     *
     */
    VMAMemoryAllocation(const SharedContext& context,
                        const std::shared_ptr<VMAMemoryAllocator>& allocator,
                        const MemoryMappingType mapping_type,
                        const VmaAllocation allocation)
        : MemoryAllocation(context), allocator(allocator), mapping_type(mapping_type),
          m_allocation(allocation) {
        SPDLOG_TRACE("create VMA allocation ({})", fmt::ptr(this));
    }

    // Unmaps and frees the memory when called
    ~VMAMemoryAllocation();

    // ------------------------------------------------------------------------------------

    void free() override;

    // ------------------------------------------------------------------------------------

    // Maps device memory to system memory.
    void* map() override;

    // Unmap memHandle
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

    VmaAllocation getAllocation() const {
        return m_allocation;
    }

    void properties(Properties& props) override;

  private:
    const std::shared_ptr<VMAMemoryAllocator> allocator;
    const MemoryMappingType mapping_type;
    VmaAllocation m_allocation;
    mutable std::mutex allocation_mutex;
    bool is_mapped = false;
};

class VMAMemoryAllocator : public MemoryAllocator {
  private:
    friend class VMAMemoryAllocation;

  public:
    static std::shared_ptr<VMAMemoryAllocator>
    // E.g. supply VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT if you want to use the device
    // address feature
    make_allocator(const SharedContext& context, const VmaAllocatorCreateFlags flags = {});

  private:
    VMAMemoryAllocator() = delete;
    explicit VMAMemoryAllocator(const SharedContext& context,
                                const VmaAllocatorCreateFlags flags = {});

  public:
    ~VMAMemoryAllocator();

    // ------------------------------------------------------------------------------------

    MemoryAllocationHandle allocate_memory(const vk::MemoryPropertyFlags required_flags,
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
