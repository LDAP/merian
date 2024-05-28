#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/memory_allocator.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

namespace merian {

Buffer::Buffer(const vk::Buffer& buffer,
               const MemoryAllocationHandle& memory,
               const vk::BufferUsageFlags& usage,
               const vk::DeviceSize& size)
    : buffer(buffer), memory(memory), usage(usage), size(size) {
    SPDLOG_TRACE("create buffer ({})", fmt::ptr(this));
}

Buffer::~Buffer() {
    SPDLOG_TRACE("destroy buffer ({})", fmt::ptr(this));
    memory->get_context()->device.destroyBuffer(buffer);
}

vk::DeviceAddress Buffer::get_device_address() {
    assert(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress);
    return memory->get_context()->device.getBufferAddress(get_buffer_device_address_info());
}

vk::BufferMemoryBarrier Buffer::buffer_barrier(const vk::AccessFlags src_access_flags,
                                               const vk::AccessFlags dst_access_flags,
                                               const vk::DeviceSize size,
                                               const uint32_t src_queue_family_index,
                                               const uint32_t dst_queue_family_index) {
    return {src_access_flags,
            dst_access_flags,
            src_queue_family_index,
            dst_queue_family_index,
            buffer,
            0,
            size};
}

vk::BufferMemoryBarrier2 Buffer::buffer_barrier2(const vk::PipelineStageFlags2 src_stage_flags,
                                                 const vk::PipelineStageFlags2 dst_stage_flags,
                                                 const vk::AccessFlags2 src_access_flags,
                                                 const vk::AccessFlags2 dst_access_flags,
                                                 const vk::DeviceSize size,
                                                 const uint32_t src_queue_family_index,
                                                 const uint32_t dst_queue_family_index) {
    return {src_stage_flags,
            src_access_flags,
            dst_stage_flags,
            dst_access_flags,
            src_queue_family_index,
            dst_queue_family_index,
            buffer,
            0,
            size,
            nullptr};
}

// --------------------------------------------------------------------------

Image::Image(const vk::Image& image,
             const MemoryAllocationHandle& memory,
             const vk::ImageCreateInfo create_info,
             const vk::ImageLayout current_layout)
    : image(image), memory(memory), create_info(create_info), current_layout(current_layout) {
    SPDLOG_TRACE("create image ({})", fmt::ptr(this));
}

Image::~Image() {
    SPDLOG_TRACE("destroy image ({})", fmt::ptr(this));
    memory->get_context()->device.destroyImage(image);
}

vk::FormatFeatureFlags Image::format_features() const {
    if (get_tiling() == vk::ImageTiling::eOptimal) {
        return memory->get_context()
            ->physical_device.physical_device.getFormatProperties(get_format())
            .optimalTilingFeatures;
    } else {
        return memory->get_context()
            ->physical_device.physical_device.getFormatProperties(get_format())
            .linearTilingFeatures;
    }
}

// Do not forget submit the barrier, else the internal state does not match the actual state
vk::ImageMemoryBarrier Image::barrier(const vk::ImageLayout new_layout,
                                      const vk::AccessFlags src_access_flags,
                                      const vk::AccessFlags dst_access_flags,
                                      const uint32_t src_queue_family_index,
                                      const uint32_t dst_queue_family_index,
                                      const vk::ImageSubresourceRange subresource_range,
                                      const bool transition_from_undefined) {
    vk::ImageLayout old_layout =
        transition_from_undefined ? vk::ImageLayout::eUndefined : current_layout;
    vk::ImageMemoryBarrier barrier{
        src_access_flags,       dst_access_flags,       old_layout, new_layout,
        src_queue_family_index, dst_queue_family_index, image,      subresource_range,
    };
    current_layout = new_layout;

    return barrier;
}

vk::ImageMemoryBarrier2 Image::barrier2(const vk::ImageLayout new_layout,
                                        const vk::AccessFlags2 src_access_flags,
                                        const vk::AccessFlags2 dst_access_flags,
                                        const vk::PipelineStageFlags2 src_stage_flags,
                                        const vk::PipelineStageFlags2 dst_stage_flags,
                                        const uint32_t src_queue_family_index,
                                        const uint32_t dst_queue_family_index,
                                        const vk::ImageSubresourceRange subresource_range,
                                        const bool transition_from_undefined) {

    vk::ImageLayout old_layout =
        transition_from_undefined ? vk::ImageLayout::eUndefined : current_layout;
    vk::ImageMemoryBarrier2 barrier{
        src_stage_flags, src_access_flags, dst_stage_flags,        dst_access_flags,
        old_layout,      new_layout,       src_queue_family_index, dst_queue_family_index,
        image,           subresource_range};
    current_layout = new_layout;

    return barrier;
}

// --------------------------------------------------------------------------

Texture::Texture(const ImageHandle& image,
                 const vk::ImageViewCreateInfo& view_create_info,
                 const SamplerHandle& sampler)
    : image(image), sampler(sampler) {
    SPDLOG_TRACE("create texture ({})", fmt::ptr(this));
    view = image->get_memory()->get_context()->device.createImageView(view_create_info);
    assert(sampler);
}

Texture::~Texture() {
    SPDLOG_TRACE("destroy texture ({})", fmt::ptr(this));
    image->get_memory()->get_context()->device.destroyImageView(view);
}

void Texture::set_sampler(const SamplerHandle& sampler) {
    assert(sampler);
    this->sampler = sampler;
}

AccelerationStructure::AccelerationStructure(
    const vk::AccelerationStructureKHR& as,
    const BufferHandle& buffer,
    const vk::AccelerationStructureBuildSizesInfoKHR& size_info)
    : as(as), buffer(buffer), size_info(size_info) {
    SPDLOG_TRACE("create acceleration structure ({})", fmt::ptr(this));
}

AccelerationStructure::~AccelerationStructure() {
    SPDLOG_TRACE("destroy acceleration structure ({})", fmt::ptr(this));
    buffer->get_memory()->get_context()->device.destroyAccelerationStructureKHR(as);
}

vk::DeviceAddress AccelerationStructure::get_acceleration_structure_device_address() {
    vk::AccelerationStructureDeviceAddressInfoKHR address_info{as};
    return buffer->get_memory()->get_context()->device.getAccelerationStructureAddressKHR(
        address_info);
}

} // namespace merian
