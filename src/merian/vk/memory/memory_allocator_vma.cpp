#include "merian/vk/memory/memory_allocator_vma.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/utils/check_result.hpp"
#include <spdlog/spdlog.h>

namespace merian {

// ALLOCATION

VMAMemoryAllocation::~VMAMemoryAllocation() {
    SPDLOG_TRACE("destroy VMA allocation ({})", fmt::ptr(this));
    unmap();
    free();
};

// ------------------------------------------------------------------------------------

void VMAMemoryAllocation::free() {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    assert(m_allocation);

    SPDLOG_TRACE("freeing memory ({})", fmt::ptr(this));
    vmaFreeMemory(allocator->vma_allocator, m_allocation);
    m_allocation = nullptr;
};

// ------------------------------------------------------------------------------------

void VMAMemoryAllocation::invalidate(const VkDeviceSize offset, const VkDeviceSize size) {
    vmaInvalidateAllocation(allocator->vma_allocator, m_allocation, offset, size);
};

void VMAMemoryAllocation::flush(const VkDeviceSize offset, const VkDeviceSize size) {
    vmaFlushAllocation(allocator->vma_allocator, m_allocation, offset, size);
};

// Maps device memory to system memory.
void* VMAMemoryAllocation::map() {
    std::lock_guard<std::mutex> lock(allocation_mutex);

    assert(m_allocation); // freed?
    assert(mapping_type != MemoryMappingType::NONE);

    if (mapped_memory != nullptr) {
        return mapped_memory;
    }

    check_result(vmaMapMemory(allocator->vma_allocator, m_allocation, &mapped_memory),
                 "mapping memory failed");
    return mapped_memory;
};

// Unmap memHandle
void VMAMemoryAllocation::unmap() {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    assert(m_allocation);

    if (mapped_memory == nullptr) {
        return;
    }

    vmaUnmapMemory(allocator->vma_allocator, m_allocation);
    mapped_memory = nullptr;
};

// ------------------------------------------------------------------------------------

ImageHandle
VMAMemoryAllocation::create_aliasing_image(const vk::ImageCreateInfo& image_create_info) {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    assert(m_allocation); // freed?

    vk::Image image;
    vmaCreateAliasingImage(allocator->vma_allocator, m_allocation,
                           reinterpret_cast<const VkImageCreateInfo*>(&image_create_info),
                           reinterpret_cast<VkImage*>(&image));

    return std::make_shared<Image>(image, shared_from_this(), image_create_info,
                                   image_create_info.initialLayout);
}

BufferHandle
VMAMemoryAllocation::create_aliasing_buffer(const vk::BufferCreateInfo& buffer_create_info) {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    assert(m_allocation); // freed?

    vk::Buffer buffer;
    vmaCreateAliasingBuffer(allocator->vma_allocator, m_allocation,
                            reinterpret_cast<const VkBufferCreateInfo*>(&buffer_create_info),
                            reinterpret_cast<VkBuffer*>(&buffer));

    return std::make_shared<Buffer>(buffer, shared_from_this(), buffer_create_info);
}

// ------------------------------------------------------------------------------------

MemoryAllocationInfo VMAMemoryAllocation::get_memory_info() const {
    const std::lock_guard<std::mutex> lock(allocation_mutex);

    VmaAllocationInfo alloc_info;
    vmaGetAllocationInfo(allocator->vma_allocator, m_allocation, &alloc_info);
    return MemoryAllocationInfo{alloc_info.deviceMemory, alloc_info.offset, alloc_info.size,
                                alloc_info.pName};
};

MemoryAllocatorHandle VMAMemoryAllocation::get_allocator() const {
    return allocator;
}

void VMAMemoryAllocation::properties(Properties& props) {
    MemoryAllocation::properties(props);

    props.output_text(fmt::format("Mapped: {}", mapped_memory != nullptr));
    if (mapped_memory != nullptr) {
        props.output_text(fmt::format("Mapped at: {}", fmt::ptr(mapped_memory)));
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------

// ALLOCATOR

/**
 * @brief      Makes an allocator.
 *
 * Factory function needed for enable_shared_from_this, see
 * https://en.cppreference.com/w/cpp/memory/enable_shared_from_this.
 */
std::shared_ptr<VMAMemoryAllocator>
VMAMemoryAllocator::make_allocator(const ContextHandle& context,
                                   const VmaAllocatorCreateFlags flags) {
    std::shared_ptr<VMAMemoryAllocator> allocator =
        std::shared_ptr<VMAMemoryAllocator>(new VMAMemoryAllocator(context, flags));
    return allocator;
}

VMAMemoryAllocator::VMAMemoryAllocator(const ContextHandle& context,
                                       const VmaAllocatorCreateFlags flags)
    : MemoryAllocator(context) {
    VmaAllocatorCreateInfo allocator_info = {.flags = flags,
                                             .physicalDevice =
                                                 context->physical_device.physical_device,
                                             .device = context->device,
                                             .preferredLargeHeapBlockSize = 0,
                                             .pAllocationCallbacks = nullptr,
                                             .pDeviceMemoryCallbacks = nullptr,
                                             .pHeapSizeLimit = nullptr,
                                             .pVulkanFunctions = nullptr,
                                             .instance = context->instance,
                                             .vulkanApiVersion = context->vk_api_version,
#if VMA_EXTERNAL_MEMORY
                                             .pTypeExternalMemoryHandleTypes = nullptr
#endif
    };
    SPDLOG_DEBUG("create VMA allocator ({})", fmt::ptr(this));
    vmaCreateAllocator(&allocator_info, &vma_allocator);
}

VMAMemoryAllocator::~VMAMemoryAllocator() {
    SPDLOG_DEBUG("destroy VMA allocator ({})", fmt::ptr(this));
    vmaDestroyAllocator(vma_allocator);
}

// ----------------------------------------------------------------------------------------------

void log_allocation([[maybe_unused]] const VmaAllocationInfo& info,
                    [[maybe_unused]] const MemoryAllocationHandle& memory,
                    [[maybe_unused]] const std::string& name) {
    if (!name.empty())
        SPDLOG_TRACE("allocated {} of memory at offset {} ({}, {})", format_size(info.size),
                     format_size(info.offset), fmt::ptr(memory.get()), name);
    else
        SPDLOG_TRACE("allocated {} of memory at offset {} ({})", format_size(info.size),
                     format_size(info.offset), fmt::ptr(memory.get()));
}

void set_name([[maybe_unused]] VmaAllocator& allocator,
              [[maybe_unused]] VmaAllocation& allocation,
              [[maybe_unused]] const std::string& name) {
#ifndef NDEBUG
    // set name for VMA leaks finder
    vmaSetAllocationName(allocator, allocation, name.c_str());
#endif
}

VmaAllocationCreateInfo make_create_info(const VmaMemoryUsage usage,
                                         const vk::MemoryPropertyFlags required_flags,
                                         const vk::MemoryPropertyFlags preferred_flags,
                                         const MemoryMappingType mapping_type,
                                         const bool dedicated,
                                         const float dedicated_priority) {
    VmaAllocationCreateInfo vma_alloc_info{
        .flags = {},
        .usage = usage,
        .requiredFlags = static_cast<VkMemoryPropertyFlags>(required_flags),
        .preferredFlags = static_cast<VkMemoryPropertyFlags>(preferred_flags),
        .memoryTypeBits = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = dedicated_priority,
    };
    // clang-format off
    vma_alloc_info.flags |= dedicated ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0;
    vma_alloc_info.flags |= mapping_type == MemoryMappingType::HOST_ACCESS_RANDOM ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : 0;
    vma_alloc_info.flags |= mapping_type == MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT : 0;
    // clang-format on
    return vma_alloc_info;
}

MemoryAllocationHandle
VMAMemoryAllocator::allocate_memory(const vk::MemoryPropertyFlags required_flags,
                                    const vk::MemoryRequirements& requirements,
                                    const std::string& debug_name,
                                    const MemoryMappingType mapping_type,
                                    const vk::MemoryPropertyFlags preferred_flags,
                                    const bool dedicated,
                                    const float dedicated_priority) {
    VmaAllocationCreateInfo vma_alloc_info =
        make_create_info(VMA_MEMORY_USAGE_UNKNOWN, required_flags, preferred_flags, mapping_type,
                         dedicated, dedicated_priority);

    VkMemoryRequirements mem_reqs = requirements;

    VmaAllocationInfo allocation_info;
    VmaAllocation allocation;
    check_result(
        vmaAllocateMemory(vma_allocator, &mem_reqs, &vma_alloc_info, &allocation, &allocation_info),
        "could not allocate memory");

    if (!debug_name.empty())
        set_name(vma_allocator, allocation, debug_name);
    const std::shared_ptr<VMAMemoryAllocator> allocator =
        static_pointer_cast<VMAMemoryAllocator>(shared_from_this());
    auto memory =
        std::make_shared<VMAMemoryAllocation>(context, allocator, mapping_type, allocation);
    log_allocation(allocation_info, memory, debug_name);
    return memory;
}

// see https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
BufferHandle VMAMemoryAllocator::create_buffer(const vk::BufferCreateInfo buffer_create_info,
                                               const MemoryMappingType mapping_type,
                                               const std::string& debug_name,
                                               const std::optional<vk::DeviceSize> min_alignment) {
    VmaAllocationCreateInfo allocation_create_info =
        make_create_info(VMA_MEMORY_USAGE_AUTO, {}, {}, mapping_type, false, 1.0);
    VkBufferCreateInfo c_buffer_create_info = buffer_create_info;

    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo allocation_info;
    if (min_alignment.has_value()) {
        check_result(vmaCreateBufferWithAlignment(vma_allocator, &c_buffer_create_info,
                                                  &allocation_create_info, min_alignment.value(),
                                                  &buffer, &allocation, &allocation_info),
                     "could not allocate memory for buffer. size == 0?");
    } else {
        check_result(vmaCreateBuffer(vma_allocator, &c_buffer_create_info, &allocation_create_info,
                                     &buffer, &allocation, &allocation_info),
                     "could not allocate memory for buffer. size == 0?");
    }
    if (!debug_name.empty())
        set_name(vma_allocator, allocation, debug_name);

    const std::shared_ptr<VMAMemoryAllocator> allocator =
        static_pointer_cast<VMAMemoryAllocator>(shared_from_this());
    auto memory =
        std::make_shared<VMAMemoryAllocation>(context, allocator, mapping_type, allocation);
    auto buffer_handle = std::make_shared<Buffer>(buffer, memory, buffer_create_info);
    log_allocation(allocation_info, memory, debug_name);

    return buffer_handle;
}

ImageHandle VMAMemoryAllocator::create_image(const vk::ImageCreateInfo image_create_info,
                                             const MemoryMappingType mapping_type,
                                             const std::string& debug_name) {
    VmaAllocationCreateInfo allocation_create_info =
        make_create_info(VMA_MEMORY_USAGE_AUTO, {}, {}, mapping_type, false, 1.0);
    VkImageCreateInfo c_image_create_info = image_create_info;

    VkImage image;
    VmaAllocation allocation;
    VmaAllocationInfo allocation_info;
    check_result(vmaCreateImage(vma_allocator, &c_image_create_info, &allocation_create_info,
                                &image, &allocation, &allocation_info),
                 "could not allocate memory for image");
    if (!debug_name.empty())
        set_name(vma_allocator, allocation, debug_name);
    const std::shared_ptr<VMAMemoryAllocator> allocator =
        static_pointer_cast<VMAMemoryAllocator>(shared_from_this());
    auto memory =
        std::make_shared<VMAMemoryAllocation>(context, allocator, mapping_type, allocation);
    auto image_handle = std::make_shared<Image>(image, memory, image_create_info);
    log_allocation(allocation_info, memory, debug_name);

    return image_handle;
}

} // namespace merian
