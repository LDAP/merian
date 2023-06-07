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

    // Return a suitable vk::BufferMemoryBarrier. Note that currently no GPU cares and a
    // global MemoryBarrier can be used in most instances without loosing performance.
    // This method is especially not very efficient because it has to call get_memory_info on the
    // memory object.
    vk::BufferMemoryBarrier
    buffer_barrier(const vk::AccessFlags src_access_flags,
                   const vk::AccessFlags dst_access_flags,
                   uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                   uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED);

  private:
    const vk::Buffer buffer;
    const MemoryAllocationHandle memory;
    const vk::BufferUsageFlags usage;
};

/**
 * @brief      Represents a vk::Image together with its memory and automatic cleanup.
 *
 * Use the transition() function to perform layout transitions to keep the internal state valid.
 */
class Image : public std::enable_shared_from_this<Image> {
  public:
    // Creats a Image objects that automatically destroys Image when destructed.
    // The memory is not freed explicitly to let it free itself.
    // It is asserted that the memory represented by `memory` is already bound correctly,
    // this is because images are commonly created by memory allocators to optimize memory accesses.
    Image(const vk::Image& image,
          const MemoryAllocationHandle& memory,
          const vk::ImageLayout current_layout = vk::ImageLayout::eUndefined);

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

    const vk::ImageLayout& get_current_layout() const {
        return current_layout;
    }

    // Use this only if you performed a layout transition without using transition_layout(...)
    // This does not perform a layout transision on itself!
    void _set_current_layout(vk::ImageLayout& new_layout) {
        current_layout = new_layout;
    }

    // Do not forget submite the barrier, else the internal state does not match the actual state
    vk::ImageMemoryBarrier
    transition_layout(const vk::ImageLayout new_layout,
                      const vk::AccessFlags src_access_flags = {},
                      const vk::AccessFlags dst_access_flags = {},
                      const uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                      const uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                      const vk::ImageAspectFlags aspect_flags = vk::ImageAspectFlagBits::eColor,
                      const uint32_t base_mip_level = 0,
                      const uint32_t mip_level_count = VK_REMAINING_MIP_LEVELS,
                      const uint32_t base_array_layer = 0,
                      const uint32_t array_layer_count = VK_REMAINING_ARRAY_LAYERS);

  private:
    const vk::Image image = VK_NULL_HANDLE;
    const MemoryAllocationHandle memory;

    vk::ImageLayout current_layout;
};

using ImageHandle = std::shared_ptr<Image>;

/**
 * @brief      A texture is a image together with a view (and subresource), and an optional sampler.
 *
 *  Try to only use the barrier() function to perform layout transitions,
 *  to keep the internal state valid.
 */
class Texture : public std::enable_shared_from_this<Texture> {
  public:
    Texture(const ImageHandle& image,
            const vk::ImageViewCreateInfo& view_create_info,
            const std::optional<SamplerHandle> sampler = std::nullopt);

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

    // Note: Can be default-initialized if no sampler is attached
    const vk::Sampler get_sampler() const {
        if (sampler.has_value())
            return *sampler.value();
        else
            return {};
    }

    // Convenience method for get_image()->get_current_layout()
    const vk::ImageLayout& get_current_layout() const {
        return image->get_current_layout();
    }

    const vk::ImageView& get_view() const {
        return view;
    }

    const vk::DescriptorImageInfo get_descriptor_info() {
        return vk::DescriptorImageInfo{get_sampler(), view, image->get_current_layout()};
    }

    void attach_sampler(const std::optional<SamplerHandle> sampler = std::nullopt);

  private:
    const ImageHandle image;
    vk::ImageView view;

    std::optional<SamplerHandle> sampler;
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
