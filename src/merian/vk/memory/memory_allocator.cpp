#include "merian/vk/memory/memory_allocator.hpp"

#include <cassert>

namespace merian {

MemoryAllocation::MemoryAllocation(const ContextHandle& context) : context(context) {}

// unmaps and frees the memory when called
MemoryAllocation::~MemoryAllocation() {}

MemoryAllocator::MemoryAllocator(const ContextHandle& context) : context(context) {
    supports_memory_requirements_without_object =
        context->vk_api_version >= VK_VERSION_1_3 ||
        context->device_extension_enabled(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
}

MemoryAllocator::~MemoryAllocator() {}

vk::MemoryRequirements
MemoryAllocator::get_image_memory_requirements(const vk::ImageCreateInfo& image_create_info) {
    if (supports_memory_requirements_without_object) {
        const vk::MemoryRequirements2 mem_req = context->device.getImageMemoryRequirements(
            vk::DeviceImageMemoryRequirements{&image_create_info});
        return mem_req.memoryRequirements;
    }

    SPDLOG_TRACE("create temporary image to query memory requirements");
    vk::Image tmp_image = context->device.createImage(image_create_info);
    vk::MemoryRequirements mem_req = context->device.getImageMemoryRequirements(tmp_image);
    context->device.destroyImage(tmp_image);
    return mem_req;
}

vk::MemoryRequirements
MemoryAllocator::get_buffer_memory_requirements(const vk::BufferCreateInfo& buffer_create_info) {
    if (supports_memory_requirements_without_object) {
        const vk::MemoryRequirements2 mem_req = context->device.getBufferMemoryRequirements(
            vk::DeviceBufferMemoryRequirements{&buffer_create_info});
        return mem_req.memoryRequirements;
    }

    SPDLOG_TRACE("create temporary image to query memory requirements");
    vk::Buffer tmp_buffer = context->device.createBuffer(buffer_create_info);
    vk::MemoryRequirements mem_req = context->device.getBufferMemoryRequirements(tmp_buffer);
    context->device.destroyBuffer(tmp_buffer);
    return mem_req;
}

BufferHandle MemoryAllocator::create_buffer(const vk::BufferCreateInfo buffer_create_info,
                                            const MemoryMappingType mapping_type,
                                            const std::string& debug_name,
                                            const std::optional<vk::DeviceSize> min_alignment) {
    vk::MemoryRequirements mem_req = get_buffer_memory_requirements(buffer_create_info);
    mem_req.alignment = std::max(mem_req.alignment, min_alignment.value_or(0));
    const auto allocation = allocate_memory({}, mem_req, debug_name, mapping_type);

    return allocation->create_aliasing_buffer(buffer_create_info);
}

ImageHandle MemoryAllocator::create_image(const vk::ImageCreateInfo image_create_info,
                                          const MemoryMappingType mapping_type,
                                          const std::string& debug_name) {

    vk::MemoryRequirements mem_req = get_image_memory_requirements(image_create_info);
    const auto allocation = allocate_memory({}, mem_req, debug_name, mapping_type);

    return allocation->create_aliasing_image(image_create_info);
}

} // namespace merian
