#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/sampler/sampler_pool.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

namespace merian {

Buffer::Buffer(const vk::Buffer& buffer,
               const MemoryAllocationHandle& memory,
               const vk::BufferUsageFlags& usage)
    : buffer(buffer), memory(memory), usage(usage) {
    SPDLOG_DEBUG("create buffer ({})", fmt::ptr(this));
}

Buffer::~Buffer() {
    SPDLOG_DEBUG("destroy buffer ({})", fmt::ptr(this));
    memory->get_context()->device.destroyBuffer(buffer);
}

vk::DeviceAddress Buffer::get_device_address() {
    assert(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress);
    return memory->get_context()->device.getBufferAddress(get_buffer_device_address_info());
}

// --------------------------------------------------------------------------

Image::Image(const vk::Image& image, const MemoryAllocationHandle& memory)
    : image(image), memory(memory) {
    SPDLOG_DEBUG("create image ({})", fmt::ptr(this));
}

Image::~Image() {
    SPDLOG_DEBUG("destroy image ({})", fmt::ptr(this));
    memory->get_context()->device.destroyImage(image);
}

// --------------------------------------------------------------------------

Texture::Texture(const ImageHandle& image,
                 const vk::DescriptorImageInfo& descriptor,
                 const std::shared_ptr<SamplerPool>& sampler_pool)
    : image(image), descriptor(descriptor), sampler_pool(sampler_pool) {
    SPDLOG_DEBUG("create texture ({})", fmt::ptr(this));
}

Texture::~Texture() {
    SPDLOG_DEBUG("destroy texture ({})", fmt::ptr(this));
    
    image->get_memory()->get_context()->device.destroyImageView(descriptor.imageView);
    if (descriptor.sampler) {
        sampler_pool->releaseSampler(descriptor.sampler);
    }
}

void Texture::attach_sampler(const vk::SamplerCreateInfo& samplerCreateInfo) {
    if (descriptor.sampler) {
        sampler_pool->releaseSampler(descriptor.sampler);
    }
    descriptor.sampler = sampler_pool->acquireSampler(samplerCreateInfo);
}

AccelerationStructure::AccelerationStructure(const vk::AccelerationStructureKHR& as,
                                             const BufferHandle& buffer)
    : as(as), buffer(buffer) {
    SPDLOG_DEBUG("create acceleration structure ({})", fmt::ptr(this));
}

AccelerationStructure::~AccelerationStructure() {
    SPDLOG_DEBUG("destroy acceleration structure ({})", fmt::ptr(this));
    buffer->get_memory()->get_context()->device.destroyAccelerationStructureKHR(as);
}

vk::DeviceAddress AccelerationStructure::get_acceleration_structure_device_address() {
    vk::AccelerationStructureDeviceAddressInfoKHR address_info{as};
    return buffer->get_memory()->get_context()->device.getAccelerationStructureAddressKHR(
        address_info);
}

} // namespace merian
