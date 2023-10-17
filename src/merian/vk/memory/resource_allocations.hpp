#pragma once

// Possible allocations together with their memory handles.

#include "merian/vk/sampler/sampler_pool.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <optional>

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
           const vk::BufferUsageFlags& usage,
           const vk::DeviceSize& size);

    ~Buffer();

    // -----------------------------------------------------------

    operator const vk::Buffer&() const noexcept {
        return buffer;
    }

    const vk::Buffer& get_buffer() const noexcept {
        return buffer;
    }

    const MemoryAllocationHandle& get_memory() const noexcept {
        return memory;
    }

    vk::DeviceSize get_size() const noexcept {
        return size;
    }

    // -----------------------------------------------------------

    vk::BufferDeviceAddressInfo get_buffer_device_address_info() {
        return vk::BufferDeviceAddressInfo{buffer};
    }

    vk::DeviceAddress get_device_address();

    // Return a suitable vk::BufferMemoryBarrier.
    [[nodiscard]] vk::BufferMemoryBarrier
    buffer_barrier(const vk::AccessFlags src_access_flags,
                   const vk::AccessFlags dst_access_flags,
                   const vk::DeviceSize size = VK_WHOLE_SIZE,
                   const uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                   const uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED);

    [[nodiscard]] vk::BufferMemoryBarrier2
    buffer_barrier2(const vk::PipelineStageFlags2 src_stage_flags,
                    const vk::PipelineStageFlags2 dst_stage_flags,
                    const vk::AccessFlags2 src_access_flags,
                    const vk::AccessFlags2 dst_access_flags,
                    const vk::DeviceSize size = VK_WHOLE_SIZE,
                    const uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
                    const uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED);

  private:
    const vk::Buffer buffer;
    const MemoryAllocationHandle memory;
    const vk::BufferUsageFlags usage;
    const vk::DeviceSize size;
};

class Image;
using ImageHandle = std::shared_ptr<Image>;

/**
 * @brief      Represents a vk::Image together with its memory and automatic cleanup.
 *
 * Use the barrier() function to perform layout transitions to keep the internal state valid.
 */
class Image : public std::enable_shared_from_this<Image> {
  public:
    // Creats a Image objects that automatically destroys Image when destructed.
    // The memory is not freed explicitly to let it free itself.
    // It is asserted that the memory represented by `memory` is already bound correctly,
    // this is because images are commonly created by memory allocators to optimize memory
    // accesses.
    Image(const vk::Image& image,
          const MemoryAllocationHandle& memory,
          const vk::Extent3D extent,
          const vk::Format format,
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

    const vk::Extent3D& get_extent() const {
        return extent;
    }

    const vk::Format& get_format() const {
        return format;
    }

    // Use this only if you performed a layout transition without using barrier(...)
    // This does not perform a layout transision on itself!
    void _set_current_layout(const vk::ImageLayout& new_layout) {
        current_layout = new_layout;
    }

    // Do not forget submite the barrier, else the internal state does not match the actual
    // state You can use transition_from_undefined when you are not interested in keeping the
    // contents, this can be more performant.
    [[nodiscard]] vk::ImageMemoryBarrier
    barrier(const vk::ImageLayout new_layout,
            const vk::AccessFlags src_access_flags = {},
            const vk::AccessFlags dst_access_flags = {},
            const uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
            const uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
            const vk::ImageSubresourceRange subresource_range = all_levels_and_layers(),
            const bool transition_from_undefined = false);

    [[nodiscard]] vk::ImageMemoryBarrier2
    barrier2(const vk::ImageLayout new_layout,
             const vk::AccessFlags2 src_access_flags = {},
             const vk::AccessFlags2 dst_access_flags = {},
             const vk::PipelineStageFlags2 src_stage_flags = {},
             const vk::PipelineStageFlags2 dst_stage_flags = {},
             const uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
             const uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
             const vk::ImageSubresourceRange subresource_range = all_levels_and_layers(),
             const bool transition_from_undefined = false);

    // If extent and range are not supplied the whole image is copied.
    // Layouts are automatically determined from get_current_layout()
    void cmd_copy_to(
        const vk::CommandBuffer& cmd,
        const ImageHandle& dst_picture,
        const std::optional<vk::Extent3D> extent = std::nullopt,
        const vk::Offset3D src_offset = {},
        const vk::Offset3D dst_offset = {},
        const std::optional<vk::ImageSubresourceLayers> opt_src_subresource = std::nullopt,
        const std::optional<vk::ImageSubresourceLayers> opt_dst_subresource = std::nullopt) {
        vk::ImageSubresourceLayers src_subresource = opt_src_subresource.value_or(first_layer());
        vk::ImageSubresourceLayers dst_subresource = opt_dst_subresource.value_or(first_layer());
        vk::ImageCopy copy{src_subresource, src_offset, dst_subresource, dst_offset,
                           extent.value_or(this->extent)};

        cmd.copyImage(image, current_layout, *dst_picture, dst_picture->get_current_layout(),
                      {copy});
    }

  private:
    const vk::Image image = VK_NULL_HANDLE;
    const MemoryAllocationHandle memory;
    const vk::Extent3D extent;
    const vk::Format format;

    vk::ImageLayout current_layout;
};

/**
 * @brief      A texture is a image together with a view (and subresource), and an optional
 * sampler.
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

    const ImageHandle& get_image() const {
        return image;
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
    AccelerationStructure(const vk::AccelerationStructureKHR& as,
                          const BufferHandle& buffer,
                          const vk::AccelerationStructureBuildSizesInfoKHR& size_info);

    ~AccelerationStructure();

    // -----------------------------------------------------------

    operator const vk::AccelerationStructureKHR&() const {
        return as;
    }

    operator const vk::AccelerationStructureKHR*() const {
        return &as;
    }

    const BufferHandle& get_buffer() const {
        return buffer;
    }

    const vk::AccelerationStructureKHR& get_acceleration_structure() const {
        return as;
    }

    const vk::AccelerationStructureBuildSizesInfoKHR& get_size_info() {
        return size_info;
    }

    // -----------------------------------------------------------

    // E.g. needed for accelerationStructureReference in VkAccelerationStructureInstanceKHR
    vk::DeviceAddress get_acceleration_structure_device_address();

  private:
    const vk::AccelerationStructureKHR as;
    const BufferHandle buffer;
    const vk::AccelerationStructureBuildSizesInfoKHR size_info;
};

using AccelerationStructureHandle = std::shared_ptr<AccelerationStructure>;

} // namespace merian
