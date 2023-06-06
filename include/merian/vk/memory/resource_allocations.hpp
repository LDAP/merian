#pragma once

// Possible allocations together with their memory handles.

#include "merian/vk/sampler/sampler_pool.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

namespace merian {

// Forward def
class MemoryAllocation;
using MemoryAllocationHandle = std::shared_ptr<MemoryAllocation>;

class Buffer : public std::enable_shared_from_this<Buffer> {
  public:
    // Creats a Buffer objects that automatically destroys buffer when destructed.
    // The memory is not freed explicitly to let it free itself.
    // It is asserted that the memory represented by `memory` is already bound to `buffer`.
    Buffer(const vk::Buffer& buffer,
           const MemoryAllocationHandle& memory,
           const vk::BufferUsageFlags& usage);

    ~Buffer();

    // -----------------------------------------------------------

    operator const vk::Buffer&() const {
        return buffer;
    }

    const vk::Buffer& get_buffer() const {
        return buffer;
    }

    const MemoryAllocationHandle& get_memory() const {
        return memory;
    }

    // -----------------------------------------------------------

    vk::BufferDeviceAddressInfo get_buffer_device_address_info() {
        return vk::BufferDeviceAddressInfo{buffer};
    }

    vk::DeviceAddress get_device_address();

  private:
    const vk::Buffer buffer;
    const MemoryAllocationHandle memory;
    const vk::BufferUsageFlags usage;
};

class Image : public std::enable_shared_from_this<Image> {
  public:
    // Creats a Image objects that automatically destroys Image when destructed.
    // The memory is not freed explicitly to let it free itself.
    // It is asserted that the memory represented by `memory` is already bound correctly.
    Image(const vk::Image& image, const MemoryAllocationHandle& memory);

    ~Image();

    // -----------------------------------------------------------

    operator const vk::Image&() const {
        return image;
    }

    const vk::Image& get_image() const {
        return image;
    }

    const MemoryAllocationHandle& get_memory() const {
        return memory;
    }

  private:
    const vk::Image image = VK_NULL_HANDLE;
    const MemoryAllocationHandle memory;
};

using ImageHandle = std::shared_ptr<Image>;

/**
 * @brief      A texture is a image together with a DescriptorImageInfo
 */
class Texture : public std::enable_shared_from_this<Texture> {
  public:
    // Creats a Texture objects that automatically destroys `image` when destructed.
    // The memory is not freed explicitly to let it free itself.
    // It is asserted that the memory is already bound correctly.
    Texture(const ImageHandle& image,
            const vk::DescriptorImageInfo& descriptor,
            const std::shared_ptr<SamplerPool>& sampler_pool);

    ~Texture();

    // -----------------------------------------------------------

    operator const vk::Image&() const {
        return *image;
    }

    const vk::Image& get_image() const {
        return *image;
    }

    const MemoryAllocationHandle& get_memory() const {
        return image->get_memory();
    }

    // Note: Can be default-initialized
    const vk::Sampler& get_sampler() const {
        return descriptor.sampler;
    }

    const vk::ImageLayout& get_layout() const {
        return descriptor.imageLayout;
    }

    const vk::ImageView& get_view() const {
        return descriptor.imageView;
    }

    const vk::DescriptorImageInfo get_descriptor_info() {
        return descriptor;
    }

    // Attaches a sampler with the vk::DescriptorImageInfo.
    // Releases any previously attached sampler.
    void attach_sampler(const vk::SamplerCreateInfo& samplerCreateInfo);

    void set_layout(const vk::ImageLayout layout) {
        descriptor.imageLayout = layout;
    }

  private:
    const ImageHandle image;
    vk::DescriptorImageInfo descriptor;
    const std::shared_ptr<SamplerPool> sampler_pool;
};

using BufferHandle = std::shared_ptr<Buffer>;
using TextureHandle = std::shared_ptr<Texture>;

class AccelerationStructure : public std::enable_shared_from_this<AccelerationStructure> {
  public:
    // Creats a AccelerationStructure objects that automatically destroys `as` when destructed.
    // The memory is not freed explicitly to let it free itself.
    // It is asserted that the memory is already bound correctly.
    AccelerationStructure(const vk::AccelerationStructureKHR& as, const BufferHandle& buffer);

    ~AccelerationStructure();

    // -----------------------------------------------------------

    operator const vk::AccelerationStructureKHR&() const {
        return as;
    }

    const BufferHandle& get_buffer() const {
        return buffer;
    }

    const vk::AccelerationStructureKHR& get_acceleration_structure() const {
        return as;
    }

    // -----------------------------------------------------------

    // E.g. needed for accelerationStructureReference in VkAccelerationStructureInstanceKHR
    vk::DeviceAddress get_acceleration_structure_device_address();

  private:
    const vk::AccelerationStructureKHR as;
    const BufferHandle buffer;
};

using AccelerationStructureHandle = std::shared_ptr<AccelerationStructure>;

} // namespace merian
