#include "merian/vk/memory/memory_allocator_vma.hpp"
#include "merian/vk/extension/extension_vma.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include <spdlog/spdlog.h>

namespace {

void log_allocation([[maybe_unused]] const VmaAllocationInfo& info,
                    [[maybe_unused]] const merian::MemoryAllocationHandle& memory,
                    [[maybe_unused]] const std::string& name) {
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_TRACE
    if (!name.empty())
        SPDLOG_TRACE("allocated {} of memory at offset {} ({}, {})", merian::format_size(info.size),
                     merian::format_size(info.offset), fmt::ptr(memory.get()), name);
    else
        SPDLOG_TRACE("allocated {} of memory at offset {} ({})", merian::format_size(info.size),
                     merian::format_size(info.offset), fmt::ptr(memory.get()));
#endif
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
                                         const merian::MemoryMappingType mapping_type,
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
    vma_alloc_info.flags |= mapping_type == merian::MemoryMappingType::HOST_ACCESS_RANDOM ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : 0;
    vma_alloc_info.flags |= mapping_type == merian::MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT : 0;
    // clang-format on
    return vma_alloc_info;
}

} // namespace

namespace merian {

// ALLOCATION

VMAMemoryAllocation::~VMAMemoryAllocation() {
    SPDLOG_TRACE("destroy VMA allocation ({})", fmt::ptr(this));

    if (map_count != 0) {
        SPDLOG_WARN(" VMA allocation ({}): unmap() must be called the same number as map()!",
                    fmt::ptr(this));

        for (uint32_t i = 0; i < map_count; i++) {
            unmap();
        }
    }

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

void* VMAMemoryAllocation::map() {
    std::lock_guard<std::mutex> lock(allocation_mutex);

    map_count++;

    if (mapped_memory != nullptr) {
        assert(map_count > 1);
        return mapped_memory;
    }

    VulkanException::throw_if_no_success(
        vmaMapMemory(allocator->vma_allocator, m_allocation, &mapped_memory),
        "mapping memory failed");
    return mapped_memory;
};

void VMAMemoryAllocation::unmap() {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    assert(map_count > 0 && "forget to call map()?");

    map_count--;
    if (map_count > 0) {
        return;
    }

    assert(mapped_memory != nullptr);

    vmaUnmapMemory(allocator->vma_allocator, m_allocation);
    mapped_memory = nullptr;
};

// ------------------------------------------------------------------------------------

ImageHandle VMAMemoryAllocation::create_aliasing_image(const vk::ImageCreateInfo& image_create_info,
                                                       const vk::DeviceSize allocation_offset) {
    std::lock_guard<std::mutex> lock(allocation_mutex);

    vk::Image image;
    vmaCreateAliasingImage2(allocator->vma_allocator, m_allocation, allocation_offset,
                            reinterpret_cast<const VkImageCreateInfo*>(&image_create_info),
                            reinterpret_cast<VkImage*>(&image));

    return Image::create(image, shared_from_this(), image_create_info,
                         image_create_info.initialLayout);
}

BufferHandle
VMAMemoryAllocation::create_aliasing_buffer(const vk::BufferCreateInfo& buffer_create_info,
                                            const vk::DeviceSize allocation_offset) {
    std::lock_guard<std::mutex> lock(allocation_mutex);

    vk::Buffer buffer;
    vmaCreateAliasingBuffer2(allocator->vma_allocator, m_allocation, allocation_offset,
                             reinterpret_cast<const VkBufferCreateInfo*>(&buffer_create_info),
                             reinterpret_cast<VkBuffer*>(&buffer));

    return Buffer::create(buffer, shared_from_this(), buffer_create_info);
}

void VMAMemoryAllocation::bind_to_image(const ImageHandle& image,
                                        const vk::DeviceSize allocation_offset) {
    vmaBindImageMemory2(allocator->vma_allocator, m_allocation, allocation_offset, **image,
                        nullptr);
    image->_set_memory_allocation(shared_from_this());
}

void VMAMemoryAllocation::bind_to_buffer(const BufferHandle& buffer,
                                         const vk::DeviceSize allocation_offset) {
    vmaBindBufferMemory2(allocator->vma_allocator, m_allocation, allocation_offset, **buffer,
                         nullptr);
    buffer->_set_memory_allocation(shared_from_this());
}

// ------------------------------------------------------------------------------------

MemoryAllocationInfo VMAMemoryAllocation::get_memory_info() const {
    const std::lock_guard<std::mutex> lock(allocation_mutex);

    VmaAllocationInfo alloc_info;
    vmaGetAllocationInfo(allocator->vma_allocator, m_allocation, &alloc_info);
    return MemoryAllocationInfo{alloc_info.deviceMemory, alloc_info.offset, alloc_info.size,
                                alloc_info.memoryType, alloc_info.pName};
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

VMAMemoryAllocator::VMAMemoryAllocator(const ContextHandle& context) : MemoryAllocator(context) {
    const auto& vma_ext = context->get_context_extension<ExtensionVMA>();

    VmaVulkanFunctions vulkan_functions = {};
    vulkan_functions.vkGetInstanceProcAddr = context->get_instance()->get_vkGetInstanceProcAddr();
    vulkan_functions.vkGetDeviceProcAddr = context->get_device()->get_vkGetDeviceProcAddr();

    VmaAllocatorCreateInfo allocator_info = {
        .flags = vma_ext->get_create_flags(),
        .physicalDevice = context->get_physical_device()->get_physical_device(),
        .device = context->get_device()->get_device(),
        .preferredLargeHeapBlockSize = 0,
        .pAllocationCallbacks = nullptr,
        .pDeviceMemoryCallbacks = nullptr,
        .pHeapSizeLimit = nullptr,
        .pVulkanFunctions = &vulkan_functions,
        .instance = context->get_instance()->get_instance(),
        .vulkanApiVersion = std::min(
            context->get_device()->get_physical_device()->get_vk_api_version(),
            VK_API_VERSION_1_4), // VMA currently asserts a maxumum version for some reason.
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
    AllocationFailed::throw_if_no_success(vmaAllocateMemory(
        vma_allocator, &mem_reqs, &vma_alloc_info, &allocation, &allocation_info));

    if (!debug_name.empty())
        set_name(vma_allocator, allocation, debug_name);
    const std::shared_ptr<VMAMemoryAllocator> allocator =
        static_pointer_cast<VMAMemoryAllocator>(shared_from_this());
    auto memory = std::make_shared<VMAMemoryAllocation>(get_context(), allocator, allocation);
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
        AllocationFailed::throw_if_no_success(
            vmaCreateBufferWithAlignment(vma_allocator, &c_buffer_create_info,
                                         &allocation_create_info, min_alignment.value(), &buffer,
                                         &allocation, &allocation_info),
            "could not allocate memory for buffer. size == 0?");
    } else {
        AllocationFailed::throw_if_no_success(vmaCreateBuffer(vma_allocator, &c_buffer_create_info,
                                                              &allocation_create_info, &buffer,
                                                              &allocation, &allocation_info),
                                              "could not allocate memory for buffer. size == 0?");
    }
    if (!debug_name.empty())
        set_name(vma_allocator, allocation, debug_name);

    const std::shared_ptr<VMAMemoryAllocator> allocator =
        static_pointer_cast<VMAMemoryAllocator>(shared_from_this());
    auto memory = std::make_shared<VMAMemoryAllocation>(get_context(), allocator, allocation);
    auto buffer_handle = Buffer::create(buffer, memory, buffer_create_info);
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
    AllocationFailed::throw_if_no_success(vmaCreateImage(vma_allocator, &c_image_create_info,
                                                         &allocation_create_info, &image,
                                                         &allocation, &allocation_info),
                                          "could not allocate memory for image");
    if (!debug_name.empty())
        set_name(vma_allocator, allocation, debug_name);
    const std::shared_ptr<VMAMemoryAllocator> allocator =
        static_pointer_cast<VMAMemoryAllocator>(shared_from_this());
    auto memory = std::make_shared<VMAMemoryAllocation>(get_context(), allocator, allocation);
    auto image_handle = Image::create(image, memory, image_create_info);
    log_allocation(allocation_info, memory, debug_name);

    return image_handle;
}

std::shared_ptr<VMAMemoryAllocator> VMAMemoryAllocator::create(const ContextHandle& context) {
    std::shared_ptr<VMAMemoryAllocator> allocator =
        std::shared_ptr<VMAMemoryAllocator>(new VMAMemoryAllocator(context));
    return allocator;
}

} // namespace merian
