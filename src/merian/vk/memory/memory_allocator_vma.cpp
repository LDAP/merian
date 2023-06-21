#include "merian/vk/memory/memory_allocator_vma.hpp"
#include "merian/utils/debug.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/utils/check_result.hpp"
#include <spdlog/spdlog.h>

namespace merian {

// ALLOCATION

VMAMemoryAllocation::~VMAMemoryAllocation() {
    SPDLOG_DEBUG("destroy VMA allocation ({})", fmt::ptr(this));
    if (is_mapped) {
        unmap();
    }
    free();
};

// ------------------------------------------------------------------------------------

void VMAMemoryAllocation::free() {
    if (m_allocation) {
        SPDLOG_DEBUG("freeing memory ({})", fmt::ptr(this));
        vmaFreeMemory(allocator->vma_allocator, m_allocation);
        m_allocation = nullptr;
    } else {
        SPDLOG_WARN("VMA allocation ({}) was already freed", fmt::ptr(this));
    }
};

// ------------------------------------------------------------------------------------

// Maps device memory to system memory.
void* VMAMemoryAllocation::map() {
    assert(m_allocation); // freed?
    assert(mapping_type != NONE);

    void* ptr;
    check_result(vmaMapMemory(allocator->vma_allocator, m_allocation, &ptr),
                 "mapping memory failed");
    is_mapped = true;
    return ptr;
};

// Unmap memHandle
void VMAMemoryAllocation::unmap() {
    assert(m_allocation); // freed?

    vmaUnmapMemory(allocator->vma_allocator, m_allocation);
    is_mapped = false;
};

// ------------------------------------------------------------------------------------

VMAMemoryAllocation::MemoryInfo VMAMemoryAllocation::get_memory_info() const {
    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(allocator->vma_allocator, m_allocation, &allocInfo);
    return MemoryInfo{allocInfo.deviceMemory, allocInfo.offset, allocInfo.size, allocInfo.pName};
};

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
VMAMemoryAllocator::make_allocator(const SharedContext& context,
                                   const VmaAllocatorCreateFlags flags) {
    std::shared_ptr<VMAMemoryAllocator> allocator =
        std::shared_ptr<VMAMemoryAllocator>(new VMAMemoryAllocator(context, flags));
    return allocator;
}

VMAMemoryAllocator::VMAMemoryAllocator(const SharedContext& context,
                                       const VmaAllocatorCreateFlags flags)
    : MemoryAllocator(context) {
    VmaAllocatorCreateInfo allocatorInfo = {
        .flags = flags,
        .physicalDevice = context->pd_container.physical_device,
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
    vmaCreateAllocator(&allocatorInfo, &vma_allocator);
}

VMAMemoryAllocator::~VMAMemoryAllocator() {
    SPDLOG_DEBUG("destroy VMA allocator ({})", fmt::ptr(this));
    vmaDestroyAllocator(vma_allocator);
}

// ----------------------------------------------------------------------------------------------

void log_allocation(const VmaAllocationInfo& info,
                    const MemoryAllocationHandle memory,
                    const std::string& name) {
    if (!name.empty())
        SPDLOG_DEBUG("allocated {} of memory at offset {} ({}, {})", format_size(info.size),
                     format_size(info.offset), fmt::ptr(memory.get()), name);
    else
        SPDLOG_DEBUG("allocated {} of memory at offset {} ({})", format_size(info.size),
                     format_size(info.offset), fmt::ptr(memory.get()));
}

void set_name(VmaAllocator& allocator, VmaAllocation& allocation, const std::string& name) {
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
    VmaAllocationCreateInfo vmaAllocInfo{
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
    vmaAllocInfo.flags |= dedicated ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0;
    vmaAllocInfo.flags |= mapping_type == HOST_ACCESS_RANDOM ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : 0;
    vmaAllocInfo.flags |= mapping_type == HOST_ACCESS_SEQUENTIAL_WRITE ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT : 0;
    // clang-format on
    return vmaAllocInfo;
}

MemoryAllocationHandle
VMAMemoryAllocator::allocate_memory(const vk::MemoryPropertyFlags required_flags,
                                    const vk::MemoryRequirements& requirements,
                                    const std::string& debug_name,
                                    const MemoryMappingType mapping_type,
                                    const vk::MemoryPropertyFlags preferred_flags,
                                    const bool dedicated,
                                    const float dedicated_priority) {
    VmaAllocationCreateInfo vmaAllocInfo =
        make_create_info(VMA_MEMORY_USAGE_UNKNOWN, required_flags, preferred_flags, mapping_type,
                         dedicated, dedicated_priority);

    VkMemoryRequirements mem_reqs = requirements;

    VmaAllocationInfo allocation_info;
    VmaAllocation allocation;
    check_result(
        vmaAllocateMemory(vma_allocator, &mem_reqs, &vmaAllocInfo, &allocation, &allocation_info),
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
        vmaCreateBufferWithAlignment(vma_allocator, &c_buffer_create_info, &allocation_create_info,
                                     min_alignment.value(), &buffer, &allocation, &allocation_info);
    } else {
        vmaCreateBuffer(vma_allocator, &c_buffer_create_info, &allocation_create_info, &buffer,
                        &allocation, &allocation_info);
    }
    if (!debug_name.empty())
        set_name(vma_allocator, allocation, debug_name);

    const std::shared_ptr<VMAMemoryAllocator> allocator =
        static_pointer_cast<VMAMemoryAllocator>(shared_from_this());
    auto memory =
        std::make_shared<VMAMemoryAllocation>(context, allocator, mapping_type, allocation);
    auto buffer_handle = std::make_shared<Buffer>(buffer, memory, buffer_create_info.usage);
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
    vmaCreateImage(vma_allocator, &c_image_create_info, &allocation_create_info, &image,
                   &allocation, &allocation_info);
    if (!debug_name.empty())
        set_name(vma_allocator, allocation, debug_name);
    const std::shared_ptr<VMAMemoryAllocator> allocator =
        static_pointer_cast<VMAMemoryAllocator>(shared_from_this());
    auto memory =
        std::make_shared<VMAMemoryAllocation>(context, allocator, mapping_type, allocation);
    auto image_handle =
        std::make_shared<Image>(image, memory, image_create_info.extent, image_create_info.format);
    log_allocation(allocation_info, memory, debug_name);

    return image_handle;
}

} // namespace merian
