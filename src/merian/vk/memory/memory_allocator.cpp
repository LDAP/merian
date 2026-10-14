#include "merian/vk/memory/memory_allocator.hpp"

#include <cassert>

namespace merian {

MemoryAllocation::MemoryAllocation(const ContextHandle& context) : context(context) {}

MemoryAllocation::~MemoryAllocation() {}

MemoryAllocator::MemoryAllocator(const ContextHandle& context)
    : context(context),
      has_maintenance4(
          context->get_device()->get_enabled_features().get_maintenance4_features().maintenance4 ==
          VK_TRUE) {}

MemoryAllocator::~MemoryAllocator() {}

vk::MemoryRequirements
MemoryAllocator::get_image_memory_requirements(const vk::ImageCreateInfo& image_create_info) {
    const vk::Device& device = context->get_device()->get_device();
    if (has_maintenance4) {
        return device
            .getImageMemoryRequirements(vk::DeviceImageMemoryRequirements{&image_create_info})
            .memoryRequirements;
    }

    SPDLOG_TRACE("create temporary image to query memory requirements");
    const vk::Image tmp_image = device.createImage(image_create_info);
    const vk::MemoryRequirements mem_req = device.getImageMemoryRequirements(tmp_image);
    device.destroyImage(tmp_image);
    return mem_req;
}

vk::MemoryRequirements
MemoryAllocator::get_buffer_memory_requirements(const vk::BufferCreateInfo& buffer_create_info) {
    const vk::Device& device = context->get_device()->get_device();
    if (has_maintenance4) {
        return device
            .getBufferMemoryRequirements(vk::DeviceBufferMemoryRequirements{&buffer_create_info})
            .memoryRequirements;
    }

    SPDLOG_TRACE("create temporary buffer to query memory requirements");
    const vk::Buffer tmp_buffer = device.createBuffer(buffer_create_info);
    const vk::MemoryRequirements mem_req = device.getBufferMemoryRequirements(tmp_buffer);
    device.destroyBuffer(tmp_buffer);
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

std::optional<uint32_t>
MemoryAllocator::find_memory_type([[maybe_unused]] const vk::BufferCreateInfo& buffer_create_info,
                                  [[maybe_unused]] const MemoryMappingType mapping_type) {
    return std::nullopt;
}

std::optional<uint32_t>
MemoryAllocator::find_memory_type([[maybe_unused]] const vk::ImageCreateInfo& image_create_info,
                                  [[maybe_unused]] const MemoryMappingType mapping_type) {
    return std::nullopt;
}

ImageHandle MemoryAllocator::create_image(const vk::ImageCreateInfo image_create_info,
                                          const MemoryMappingType mapping_type,
                                          const std::string& debug_name) {

    vk::MemoryRequirements mem_req = get_image_memory_requirements(image_create_info);
    const auto allocation = allocate_memory({}, mem_req, debug_name, mapping_type);

    return allocation->create_aliasing_image(image_create_info);
}

} // namespace merian
