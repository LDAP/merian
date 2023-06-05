#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <memory>
#include <vulkan/vulkan.hpp>

#include <string>

namespace merian {

// Forward def
class MemoryAllocation;
using MemoryAllocationHandle = std::shared_ptr<MemoryAllocation>;

static const MemoryAllocationHandle NullMememoryAllocationHandle = nullptr;

enum MemoryMappingType {
    // Memory mapping is not possible. GPU-only resources.
    // Will likely have VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    NONE,
    // Memory mapping is possible. Memory can be access randomly.
    // Equals VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT from VMA.
    // Will allways have VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT and VK_MEMORY_PROPERTY_HOST_CACHED_BIT
    HOST_ACCESS_RANDOM,
    // Memory mapping is possible. Memory can only be access sequentially (memcpy, for-loop).
    // Equals VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT from VMA.
    // Eg. for a staging buffer for upload.
    // Will always have VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    HOST_ACCESS_SEQUENTIAL_WRITE
};

/**
 * merian::MemAllocator is a Vulkan memory allocator interface extensively used by
 * ResourceAllocator. It provides means to allocate, free, map and unmap pieces of Vulkan device
 * memory. Concrete implementations derive from merian::MemoryAllocator. They can implement the
 * allocator dunctionality themselves or act as an adapter to another memory allocator
 * implementation.
 *
 * A merian::MemAllocator hands out opaque 'MemHandles'. The implementation of the MemAllocator
 * interface may chose any type of payload to store in a MemHandle. A MemHandle's relevant
 * information can be retrieved via getMemoryInfo().
 *
 * If you want to map memory use the methods directly on the MemoryAllocation.
 */
class MemoryAllocator : public std::enable_shared_from_this<MemoryAllocator> {
  public:
    MemoryAllocator(const SharedContext& context) : context(context) {}

    // Make sure the dtor is virtual
    virtual ~MemoryAllocator() = default;

  public:
    // Direct highly discouraged. Use create_buffer and create_image instead.
    virtual MemoryAllocationHandle
    allocate_memory(const vk::MemoryPropertyFlags required_flags,
                    const vk::MemoryRequirements& requirements,
                    const std::string& debug_name = {},
                    const MemoryMappingType mapping_type = NONE,
                    const vk::MemoryPropertyFlags preferred_flags = {},
                    const bool dedicated = false,
                    const float dedicated_priority = 1.0) = 0;

    virtual BufferHandle
    create_buffer(const vk::BufferCreateInfo buffer_create_info,
                  const MemoryMappingType mapping_type = NONE,
                  const std::string& debug_name = {},
                  const std::optional<vk::DeviceSize> min_alignment = std::nullopt) = 0;

    virtual ImageHandle create_image(const vk::ImageCreateInfo image_create_info,
                                     const MemoryMappingType mapping_type = NONE,
                                     const std::string& debug_name = {}) = 0;

    // ------------------------------------------------------------------------------------

  public:
    const SharedContext& get_context() {
        return context;
    }

  protected:
    const SharedContext context;
};

/* MemoryAllocation represents a memory allocation or sub-allocation from the
 * generic merian::MemoryAllocator interface. Ideally use `merian::NullMememoryAllocationHandle` for
 * setting to 'NULL'.
 *
 * Base class for memory handles, individual allocators will derive from it and fill the handles
 * with their own data.
 */
class MemoryAllocation : public std::enable_shared_from_this<MemoryAllocation> {
  public:
    struct MemoryInfo {
        MemoryInfo(const vk::DeviceMemory memory,
                   const vk::DeviceSize offset,
                   const vk::DeviceSize size,
                   const char* name)
            : memory(memory), offset(offset), size(size), name(name) {}

        const vk::DeviceMemory memory;
        const vk::DeviceSize offset;
        const vk::DeviceSize size;
        const char* name;
    };

  public:
    MemoryAllocation(const SharedContext& context) : context(context) {}

    // unmaps and frees the memory when called
    virtual ~MemoryAllocation() = default;

    // ------------------------------------------------------------------------------------

    // can only be called once
    virtual void free() = 0;

    // ------------------------------------------------------------------------------------

    // Convenience function to allow mapping straight to a typed pointer.
    template <class T> T* map_as() {
        return (T*)map();
    }

    // Maps device memory to system memory.
    virtual void* map() = 0;

    // Unmap memHandle
    virtual void unmap() = 0;

    // ------------------------------------------------------------------------------------

    // Retrieve detailed information about 'memHandle'
    virtual MemoryInfo get_memory_info() const = 0;

    const SharedContext& get_context() const {
        return context;
    }

  protected:
    const SharedContext context;
};

} // namespace merian
