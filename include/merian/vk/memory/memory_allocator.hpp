#pragma once

#include "merian/utils/string.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <memory>
#include <vulkan/vulkan.hpp>

#include <optional>
#include <string>

namespace merian {

// Forward def
class MemoryAllocation;
using MemoryAllocationHandle = std::shared_ptr<MemoryAllocation>;
class MemoryAllocator;
using MemoryAllocatorHandle = std::shared_ptr<MemoryAllocator>;

static const MemoryAllocationHandle NULL_MEMEMORY_ALLOCATION_HANDLE = nullptr;

enum class MemoryMappingType {
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
    MemoryAllocator(const ContextHandle& context) : context(context) {}

    // Make sure the dtor is virtual
    virtual ~MemoryAllocator() = default;

  public:
    // Direct highly discouraged. Use create_buffer and create_image instead.
    virtual MemoryAllocationHandle
    allocate_memory(const vk::MemoryPropertyFlags required_flags,
                    const vk::MemoryRequirements& requirements,
                    const std::string& debug_name = {},
                    const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                    const vk::MemoryPropertyFlags preferred_flags = {},
                    const bool dedicated = false,
                    const float dedicated_priority = 1.0) = 0;

    virtual BufferHandle
    create_buffer(const vk::BufferCreateInfo buffer_create_info,
                  const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                  const std::string& debug_name = {},
                  const std::optional<vk::DeviceSize> min_alignment = std::nullopt) = 0;

    virtual ImageHandle create_image(const vk::ImageCreateInfo image_create_info,
                                     const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                                     const std::string& debug_name = {}) = 0;

    // ------------------------------------------------------------------------------------

  public:
    const ContextHandle& get_context() {
        return context;
    }

  private:
    const ContextHandle context;
};

struct MemoryAllocationInfo {
    MemoryAllocationInfo(const vk::DeviceMemory memory,
                         const vk::DeviceSize offset,
                         const vk::DeviceSize size,
                         const char* name)
        : memory(memory), offset(offset), size(size), name(name) {}

    const vk::DeviceMemory memory;
    const vk::DeviceSize offset;
    const vk::DeviceSize size;
    const char* name;
};

inline std::string format_as(const MemoryAllocationInfo& alloc_info) {
    return fmt::format("DeviceMemory: {}\nOffset: {}\nSize: {}\nName: {}",
                       fmt::ptr(static_cast<VkDeviceMemory>(alloc_info.memory)),
                       format_size(alloc_info.offset), format_size(alloc_info.size),
                       (alloc_info.name != nullptr) ? alloc_info.name : "<unknown>");
}

/* MemoryAllocation represents a memory allocation or sub-allocation from the
 * generic merian::MemoryAllocator interface. Ideally use `merian::NullMememoryAllocationHandle` for
 * setting to 'NULL'.
 *
 * Base class for memory handles, individual allocators will derive from it and fill the handles
 * with their own data.
 */
class MemoryAllocation : public std::enable_shared_from_this<MemoryAllocation> {
  public:
    MemoryAllocation(const ContextHandle& context);

    // unmaps and frees the memory when called
    virtual ~MemoryAllocation();

    // ------------------------------------------------------------------------------------

    // Convenience function to allow mapping straight to a typed pointer.
    template <class T> T* map_as() {
        return (T*)map();
    }

    // Invalidates memory of a allocation. Call this before reading from non host-coherent memory
    // type or before reading from persistently mapped host-coherent memory.
    // Map does not do that automatically, internally this is a call to
    // vkInvalidateMappedMemoryRanges
    virtual void invalidate([[maybe_unused]] const VkDeviceSize offset = 0,
                            [[maybe_unused]] const VkDeviceSize size = VK_WHOLE_SIZE) {
        throw std::runtime_error{"invalidate is unsupported for this memory type"};
    }

    // Call this after writing to non host-coherent memory
    // or after writing to persistently mapped host-coherent memory.
    // Map does not do that automatically, internally this is a call to vkFlushMappedMemoryRanges
    virtual void flush([[maybe_unused]] const VkDeviceSize offset = 0,
                       [[maybe_unused]] const VkDeviceSize size = VK_WHOLE_SIZE) {
        throw std::runtime_error{"flush is unsupported for this memory type"};
    }

    // Maps device memory to system memory. This should return the same pointer if called multiple
    // times before "unmap".
    virtual void* map() {
        throw std::runtime_error{"mapping is unsupported for this memory type"};
    };

    // Unmap memHandle
    virtual void unmap() {
        throw std::runtime_error{"mapping is unsupported for this memory type"};
    }

    // ------------------------------------------------------------------------------------

    // Creates an image that points to this memory
    virtual ImageHandle
    create_aliasing_image([[maybe_unused]] const vk::ImageCreateInfo& image_create_info) {
        throw std::runtime_error{"create aliasing images is unsupported for this memory type"};
    }

    // Creates a buffer that points to this memory
    virtual BufferHandle
    create_aliasing_buffer([[maybe_unused]] const vk::BufferCreateInfo& buffer_create_info) {
        throw std::runtime_error{"create aliasing images is unsupported for this memory type"};
    }

    // ------------------------------------------------------------------------------------

    // Retrieve detailed information about 'memHandle'. This may not be very efficient.
    // Try to avoid if possible
    virtual MemoryAllocationInfo get_memory_info() const = 0;

    virtual MemoryAllocatorHandle get_allocator() const = 0;

    const ContextHandle& get_context() const {
        return context;
    }

    virtual void properties(Properties& props) {
        props.output_text(fmt::format("{}", get_memory_info()));
    }

  private:
    const ContextHandle context;
};

} // namespace merian
