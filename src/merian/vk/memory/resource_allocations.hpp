#pragma once

// Possible allocations together with their memory handles.

#include "merian/utils/properties.hpp"
#include "merian/vk/sampler/sampler.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <optional>

namespace merian {

// Forward def
class MemoryAllocation;
using MemoryAllocationHandle = std::shared_ptr<MemoryAllocation>;
class Buffer;
using BufferHandle = std::shared_ptr<Buffer>;

class Buffer : public std::enable_shared_from_this<Buffer> {

  public:
    constexpr static vk::BufferUsageFlags SCRATCH_BUFFER_USAGE =
        vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer;

    constexpr static vk::BufferUsageFlags INSTANCES_BUFFER_USAGE =
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress |
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;

  public:
    // Creats a Buffer objects that automatically destroys buffer when destructed.
    // The memory is not freed explicitly to let it free itself.
    // It is asserted that the memory represented by `memory` is already bound to `buffer`.
    Buffer(const vk::Buffer& buffer,
           const MemoryAllocationHandle& memory,
           const vk::BufferCreateInfo& create_info);

    ~Buffer();

    // -----------------------------------------------------------

    operator const vk::Buffer&() const noexcept {
        return buffer;
    }

    const vk::Buffer& get_buffer() const noexcept {
        return buffer;
    }

    const vk::Buffer& operator*() {
        return buffer;
    }

    const MemoryAllocationHandle& get_memory() const noexcept {
        return memory;
    }

    vk::DeviceSize get_size() const noexcept {
        return create_info.size;
    }

    // -----------------------------------------------------------

    vk::BufferDeviceAddressInfo get_buffer_device_address_info() {
        return vk::BufferDeviceAddressInfo{buffer};
    }

    vk::DeviceAddress get_device_address();

    // Remember to add an buffer barrier to the command buffer.
    void fill(const vk::CommandBuffer& cmd, const uint32_t data = 0) {
        cmd.fillBuffer(buffer, 0, VK_WHOLE_SIZE, data);
    }

    BufferHandle create_aliasing_buffer();

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

    // -----------------------------------------------------------

    void properties(Properties& props);

  private:
    const vk::Buffer buffer;
    const MemoryAllocationHandle memory;
    const vk::BufferCreateInfo create_info;
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
    // Creates a Image objects that automatically destroys Image when destructed.
    // The memory is not freed explicitly to let it free itself.
    // It is asserted that the memory represented by `memory` is already bound correctly,
    // this is because images are commonly created by memory allocators to optimize memory
    // accesses.
    Image(const vk::Image& image,
          const MemoryAllocationHandle& memory,
          const vk::ImageCreateInfo create_info,
          const vk::ImageLayout current_layout = vk::ImageLayout::eUndefined);

    ~Image();

    // -----------------------------------------------------------

    operator const vk::Image&() const {
        return image;
    }

    const vk::Image& get_image() const {
        return image;
    }

    const vk::Image& operator*() {
        return image;
    }

    const MemoryAllocationHandle& get_memory() const {
        return memory;
    }

    const vk::ImageLayout& get_current_layout() const {
        return current_layout;
    }

    const vk::Extent3D& get_extent() const {
        return create_info.extent;
    }

    const vk::Format& get_format() const {
        return create_info.format;
    }

    const vk::ImageTiling& get_tiling() const {
        return create_info.tiling;
    }

    const vk::ImageUsageFlags& get_usage_flags() const {
        return create_info.usage;
    }

    // Use this only if you performed a layout transition without using barrier(...)
    // This does not perform a layout transision on itself!
    void _set_current_layout(const vk::ImageLayout& new_layout) {
        current_layout = new_layout;
    }

    // Guess AccessFlags from old and new layout.
    [[nodiscard]] vk::ImageMemoryBarrier barrier(const vk::ImageLayout new_layout);

    // Guess AccessFlags2 and PipelineStageFlags2 from old and new layout.
    [[nodiscard]] vk::ImageMemoryBarrier2 barrier2(const vk::ImageLayout new_layout);

    // Do not forget submite the barrier, else the internal state does not match the actual
    // state You can use transition_from_undefined when you are not interested in keeping the
    // contents, this can be more performant.
    [[nodiscard]] vk::ImageMemoryBarrier
    barrier(const vk::ImageLayout new_layout,
            const vk::AccessFlags src_access_flags,
            const vk::AccessFlags dst_access_flags,
            const uint32_t src_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
            const uint32_t dst_queue_family_index = VK_QUEUE_FAMILY_IGNORED,
            const vk::ImageSubresourceRange subresource_range = all_levels_and_layers(),
            const bool transition_from_undefined = false);

    [[nodiscard]] vk::ImageMemoryBarrier2
    barrier2(const vk::ImageLayout new_layout,
             const vk::AccessFlags2 src_access_flags,
             const vk::AccessFlags2 dst_access_flags,
             const vk::PipelineStageFlags2 src_stage_flags,
             const vk::PipelineStageFlags2 dst_stage_flags,
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
                           extent.value_or(this->get_extent())};

        cmd.copyImage(image, current_layout, *dst_picture, dst_picture->get_current_layout(),
                      {copy});
    }

    // Convenience method to create a view info.
    // By default all levels and layers are accessed and if array layers > 1 a array view is used.
    // If the image is 2D and is_cube is true a cube view is returned.
    vk::ImageViewCreateInfo make_view_create_info(const bool is_cube = false) {
        vk::ImageViewCreateInfo view_info{
            {}, get_image(), {}, create_info.format, {}, all_levels_and_layers(),
        };

        switch (create_info.imageType) {
        case vk::ImageType::e1D:
            view_info.viewType = (create_info.arrayLayers > 1 ? vk::ImageViewType::e1DArray
                                                              : vk::ImageViewType::e1D);
            break;
        case vk::ImageType::e2D:
            if (is_cube) {
                view_info.viewType = vk::ImageViewType::eCube;
            } else {
                view_info.viewType = create_info.arrayLayers > 1 ? vk::ImageViewType::e2DArray
                                                                 : vk::ImageViewType::e2D;
            }
            break;
        case vk::ImageType::e3D:
            view_info.viewType = vk::ImageViewType::e3D;
            break;
        default:
            assert(0);
        }

        return view_info;
    }

    vk::FormatFeatureFlags format_features() const;

    // Test if the image has been created with a usage value containing at least one of the usages
    // defined in the valid image usage list for image views
    // (https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageViewCreateInfo-image-04441)
    bool valid_for_view();

    ImageHandle create_aliasing_image();

    // -----------------------------------------------------------

    void properties(Properties& props);

  private:
    const vk::Image image = VK_NULL_HANDLE;
    const MemoryAllocationHandle memory;
    const vk::ImageCreateInfo create_info;

    vk::ImageLayout current_layout;
};

/**
 * @brief      A texture is a image together with a view (and subresource) and sampler.
 *
 *  Try to only use the barrier() function to perform layout transitions,
 *  to keep the internal state valid.
 */
class Texture : public std::enable_shared_from_this<Texture> {
  public:
    Texture(const vk::ImageView& view, const ImageHandle& image, const SamplerHandle& sampler);

    ~Texture();

    // -----------------------------------------------------------

    operator const vk::Image&() const {
        return *image;
    }

    operator const vk::ImageView&() const {
        return view;
    }

    const vk::ImageView& operator*() {
        return view;
    }

    const ImageHandle& get_image() const {
        return image;
    }

    const MemoryAllocationHandle& get_memory() const {
        return image->get_memory();
    }

    const SamplerHandle get_sampler() const {
        return sampler;
    }

    // Convenience method for get_image()->get_current_layout()
    const vk::ImageLayout& get_current_layout() const {
        return image->get_current_layout();
    }

    const vk::ImageView& get_view() const {
        return view;
    }

    const vk::DescriptorImageInfo get_descriptor_info() {
        return vk::DescriptorImageInfo{*get_sampler(), view, image->get_current_layout()};
    }

    void set_sampler(const SamplerHandle& sampler);

    // -----------------------------------------------------------

    void properties(Properties& props);

  private:
    const vk::ImageView view;
    const ImageHandle image;
    SamplerHandle sampler;
};

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

    const vk::AccelerationStructureKHR& operator*() {
        return as;
    }

    const BufferHandle& get_buffer() const {
        return buffer;
    }

    const vk::AccelerationStructureKHR& get_acceleration_structure() const {
        return as;
    }

    const vk::AccelerationStructureBuildSizesInfoKHR& get_size_info() const {
        return size_info;
    }

    // -----------------------------------------------------------

    // E.g. needed for accelerationStructureReference in VkAccelerationStructureInstanceKHR
    vk::DeviceAddress get_acceleration_structure_device_address();

    // A barrier to insert between tlas builds and tlas usage.
    vk::BufferMemoryBarrier2 tlas_read_barrier2(const vk::PipelineStageFlags2 read_stages) const {
        return buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                       read_stages,
                                       vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
                                       vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    }

    // A barier to insert between blas builds and blas usage.
    vk::BufferMemoryBarrier2 blas_read_barrier2() const {
        return buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                       vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                       vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
                                       vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    }

    // A barier to insert between blas builds and blas usage.
    vk::BufferMemoryBarrier blas_read_barrier() const {
        return buffer->buffer_barrier(vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                      vk::AccessFlagBits::eAccelerationStructureReadKHR);
    }

    // A barrier to insert between tlas usage and tlas rebuild/update.
    vk::BufferMemoryBarrier2 tlas_build_barrier2(const vk::PipelineStageFlags2 read_stages) const {
        return buffer->buffer_barrier2(read_stages,
                                       vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                       vk::AccessFlagBits2::eAccelerationStructureReadKHR,
                                       vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
    }

    // A barier to insert between blas read (for TLAS build) and blas rebuild/update.
    vk::BufferMemoryBarrier2 blas_build_barrier2() const {
        return buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                       vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                                       vk::AccessFlagBits2::eAccelerationStructureReadKHR,
                                       vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
    }

    // A barier to insert between blas usage and blas rebuild/update.
    vk::BufferMemoryBarrier blas_build_barrier() const {
        return buffer->buffer_barrier(vk::AccessFlagBits::eAccelerationStructureReadKHR,
                                      vk::AccessFlagBits::eAccelerationStructureWriteKHR);
    }

    // -----------------------------------------------------------

    void properties(Properties& props);

  private:
    const vk::AccelerationStructureKHR as;
    const BufferHandle buffer;
    const vk::AccelerationStructureBuildSizesInfoKHR size_info;
};

using AccelerationStructureHandle = std::shared_ptr<AccelerationStructure>;

} // namespace merian
