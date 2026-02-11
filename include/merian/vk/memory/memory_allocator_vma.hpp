#pragma once

#include "merian/vk/memory/memory_allocator.hpp"

#include <cstdio>
#include <optional>
#include <spdlog/spdlog.h>
#include <vk_mem_alloc.h>

namespace merian {

class VMAMemoryAllocator;
using VMAMemoryAllocatorHandle = std::shared_ptr<VMAMemoryAllocator>;

class VMAMemoryAllocation : public MemoryAllocation {
  public:
    VMAMemoryAllocation() = delete;
    VMAMemoryAllocation(const VMAMemoryAllocation&) = delete;
    VMAMemoryAllocation(VMAMemoryAllocation&&) = delete;

    /**
     * @brief      Constructs a new instance.
     *
     */
    VMAMemoryAllocation(const ContextHandle& context,
                        const std::shared_ptr<VMAMemoryAllocator>& allocator,
                        VmaAllocation allocation)
        : MemoryAllocation(context), allocator(allocator), m_allocation(allocation) {
        SPDLOG_TRACE("create VMA allocation ({})", fmt::ptr(this));
    }

    // frees the memory when called
    ~VMAMemoryAllocation();

    // ------------------------------------------------------------------------------------

    void invalidate(const VkDeviceSize offset = 0,
                    const VkDeviceSize size = VK_WHOLE_SIZE) override;

    void flush(const VkDeviceSize offset = 0, const VkDeviceSize size = VK_WHOLE_SIZE) override;

    // You must call unmap the same number of time you call map!
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

    VmaAllocation get_allocation() const {
        return m_allocation;
    }

    void properties(Properties& props) override;

  private:
    const std::shared_ptr<VMAMemoryAllocator> allocator;
    VmaAllocation m_allocation;

    mutable std::mutex allocation_mutex;
    void* mapped_memory = nullptr;
    uint32_t map_count = 0;
};

/**
 * @brief      A memory allocator using VulkanMemoryAllocator. Needs the merian-vma extension to be
 * enabled.
 */
class VMAMemoryAllocator : public MemoryAllocator {
  private:
    friend class VMAMemoryAllocation;

    VMAMemoryAllocator() = delete;
    explicit VMAMemoryAllocator(const ContextHandle& context);

  public:
    ~VMAMemoryAllocator();

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

  public:
    static std::shared_ptr<VMAMemoryAllocator> create(const ContextHandle& context);
};

} // namespace merian
