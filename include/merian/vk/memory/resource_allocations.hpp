#pragma once

// Possible allocations together with their memory handles.

#include "merian/utils/properties.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/object.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

#include <optional>

namespace merian {

class Resource : public Object {
  public:
    virtual ~Resource();
};
using ResourceHandle = std::shared_ptr<Resource>;

// Forward def
class MemoryAllocation;
using MemoryAllocationHandle = std::shared_ptr<MemoryAllocation>;
class Buffer;
using BufferHandle = std::shared_ptr<Buffer>;

/**
 * @brief      This class describes a vk::Sampler with automatic cleanup.
 */
class Sampler : public std::enable_shared_from_this<Sampler>, public Resource {
  public:
    Sampler(const ContextHandle& context, const vk::SamplerCreateInfo& create_info);

    ~Sampler();

    operator const vk::Sampler&() const {
        return sampler;
    }

    const vk::Sampler& operator*() {
        return sampler;
    }

    const vk::Sampler& get_sampler() const {
        return sampler;
    }

    vk::DescriptorImageInfo get_descriptor_info() const;

  private:
    const ContextHandle context;
    vk::Sampler sampler;
};
using SamplerHandle = std::shared_ptr<Sampler>;

class Buffer : public std::enable_shared_from_this<Buffer>, public Resource {

  public:
    constexpr static vk::BufferUsageFlags SCRATCH_BUFFER_USAGE =
        vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer;

    constexpr static vk::BufferUsageFlags INSTANCES_BUFFER_USAGE =
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress |
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;

  protected:
    // Creates a Buffer objects that automatically destroys buffer when destructed.
    // The memory is not freed explicitly to let it free itself.
    // It is asserted that the memory represented by `memory` is already bound to `buffer`.
    Buffer(const vk::Buffer& buffer,
           const MemoryAllocationHandle& memory,
           const vk::BufferCreateInfo& create_info);

    Buffer(const ContextHandle& context, const vk::BufferCreateInfo& create_info);

  public:
    ~Buffer();

    // -----------------------------------------------------------

    operator const vk::Buffer&() const noexcept {
        return buffer;
    }

    const vk::Buffer& get_buffer() const noexcept {
        return buffer;
    }

    const vk::Buffer& operator*() const noexcept {
        return buffer;
    }

    // returns nullptr if not bound to memory
    const MemoryAllocationHandle& get_memory() const noexcept {
        return memory;
    }

    vk::DeviceSize get_size() const noexcept {
        return create_info.size;
    }

    const ContextHandle& get_context() const {
        return context;
    }

    const vk::BufferUsageFlags& get_usage_flags() const {
        return create_info.usage;
    }

    // -----------------------------------------------------------

    vk::DescriptorBufferInfo get_descriptor_info(const vk::DeviceSize offset = 0,
                                                 const vk::DeviceSize range = VK_WHOLE_SIZE) const;

    vk::DescriptorAddressInfoEXT
    get_descriptor_address_info(const vk::DeviceSize offset = 0,
                                const vk::DeviceSize range = VK_WHOLE_SIZE) const;

    vk::BufferDeviceAddressInfo get_buffer_device_address_info() const;

    vk::MemoryRequirements get_memory_requirements() const;

    vk::DeviceAddress get_device_address() const;

    BufferHandle create_aliasing_buffer() const;

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

    // Use only by memory allocators to set after memory was bound to this resource.
    void _set_memory_allocation(const MemoryAllocationHandle& allocation);

    void properties(Properties& props);

  private:
    const ContextHandle context;
    const vk::Buffer buffer;
    MemoryAllocationHandle memory;
    const vk::BufferCreateInfo create_info;

  public:
    static inline const BufferHandle EMPTY = nullptr;

    static BufferHandle create(const vk::Buffer& buffer,
                               const MemoryAllocationHandle& memory,
                               const vk::BufferCreateInfo& create_info);

    static BufferHandle create(const ContextHandle& context,
                               const vk::BufferCreateInfo& create_info);
};

class Image;
using ImageHandle = std::shared_ptr<Image>;

/**
 * @brief      Represents a vk::Image together with its memory and automatic cleanup.
 *
 * Use the barrier() function to perform layout transitions to keep the internal state valid.
 */
class Image : public std::enable_shared_from_this<Image>, public Resource {

  protected:
    // It is asserted that the memory represented by `memory` is already bound correctly,
    // this is because images are commonly created by memory allocators to optimize memory
    // accesses.
    Image(const vk::Image& image,
          const MemoryAllocationHandle& memory,
          const vk::ImageCreateInfo create_info,
          const vk::ImageLayout current_layout = vk::ImageLayout::eUndefined);

    // Create an image that is not bound to memory.
    Image(const ContextHandle& context,
          const vk::Image& image,
          const vk::ImageCreateInfo create_info,
          const vk::ImageLayout current_layout = vk::ImageLayout::eUndefined);

    // Create an image that is not bound to memory.
    Image(const ContextHandle& context, const vk::ImageCreateInfo create_info);

  public:
    virtual ~Image();

    // -----------------------------------------------------------

    operator const vk::Image&() const {
        return image;
    }

    const vk::Image& get_image() const {
        return image;
    }

    const vk::Image& operator*() const {
        return image;
    }

    // returns nullptr if not bound to memory
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

    const ContextHandle& get_context() const {
        return context;
    }

    // Use this only if you performed a layout transition without using barrier(...)
    // This does not perform a layout transision on itself!
    void _set_current_layout(const vk::ImageLayout& new_layout) {
        current_layout = new_layout;
    }

    // Use only by memory allocators to set after memory was bound to this resource.
    void _set_memory_allocation(const MemoryAllocationHandle& allocation);

    // Guess AccessFlags from old and new layout.
    [[nodiscard]] vk::ImageMemoryBarrier barrier(const vk::ImageLayout new_layout,
                                                 const bool transition_from_undefined = false);

    // Guess AccessFlags2 and PipelineStageFlags2 from old and new layout.
    [[nodiscard]] vk::ImageMemoryBarrier2 barrier2(const vk::ImageLayout new_layout,
                                                   const bool transition_from_undefined = false);

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

    // Convenience method to create a view info.
    // By default all levels and layers are accessed and if array layers > 1 a array view is used.
    // If the image is 2D and is_cube is true a cube view is returned.
    vk::ImageViewCreateInfo make_view_create_info(const bool is_cube = false) const;

    vk::MemoryRequirements get_memory_requirements() const;

    vk::FormatFeatureFlags format_features() const;

    // Test if the image has been created with a usage value containing at least one of the usages
    // defined in the valid image usage list for image views
    // (https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkImageViewCreateInfo-image-04441)
    bool valid_for_view();

    ImageHandle create_aliasing_image();

    // -----------------------------------------------------------

    void properties(Properties& props);

  protected:
    vk::Image& get_image() {
        return image;
    }

  private:
    const ContextHandle context;
    vk::Image image;
    MemoryAllocationHandle memory;
    const vk::ImageCreateInfo create_info;

    vk::ImageLayout current_layout;

  public:
    static inline const ImageHandle EMPTY = nullptr;

    static bool valid_for_view(const vk::ImageUsageFlags usage_flags);

    static ImageHandle create(const vk::Image& image,
                              const MemoryAllocationHandle& memory,
                              const vk::ImageCreateInfo create_info,
                              const vk::ImageLayout current_layout = vk::ImageLayout::eUndefined);

    static ImageHandle create(const ContextHandle& context,
                              const vk::Image& image,
                              const vk::ImageCreateInfo create_info,
                              const vk::ImageLayout current_layout = vk::ImageLayout::eUndefined);

    static ImageHandle create(const ContextHandle& context, const vk::ImageCreateInfo create_info);

    // Returns the size in bytes for a pixel of this format. Supported are only non-block compressed
    // formats.
    static vk::DeviceSize format_size(const vk::Format format);
};

class ImageView;
using ImageViewHandle = std::shared_ptr<ImageView>;

/**
 * @brief An wrapper for vk::ImageViews
 *
 *  Try to only use the barrier() function to perform layout transitions,
 *  to keep the internal state valid.
 */
class ImageView : public std::enable_shared_from_this<ImageView>, public Resource {

  protected:
    ImageView(const vk::ImageViewCreateInfo& view_create_info, const ImageHandle& image);

    ImageView(const vk::ImageView& view, const ImageHandle& image);

  public:
    ~ImageView();

    // -----------------------------------------------------------

    operator const vk::ImageView&() const {
        return view;
    }

    const vk::ImageView& operator*() const {
        return view;
    }

    const vk::ImageView& get_view() const {
        return view;
    }

    vk::DescriptorImageInfo
    get_descriptor_info(const std::optional<vk::ImageLayout> access_layout = std::nullopt) const;

    // -----------------------------------------------------------

    operator const vk::Image&() const {
        return *image;
    }

    const ImageHandle& get_image() const {
        return image;
    }

    // -----------------------------------------------------------

    void properties(Properties& props);

  private:
    vk::ImageView view;
    const ImageHandle image;

  public:
    static inline const ImageViewHandle EMPTY = nullptr;

    static ImageViewHandle create(const vk::ImageViewCreateInfo& view_create_info,
                                  const ImageHandle& image);

    static ImageViewHandle create(const ImageHandle& image);

    static ImageViewHandle create(const vk::ImageView& view, const ImageHandle& image);
};

class Texture;
using TextureHandle = std::shared_ptr<Texture>;

/**
 * @brief      A texture is an ImageView with Sampler, i.e. what is needed to create a descriptor.
 *
 *  Try to only use the barrier() function to perform layout transitions,
 *  to keep the internal state valid.
 */
class Texture : public std::enable_shared_from_this<Texture>, public Resource {
  protected:
    Texture(const ImageViewHandle& view, const SamplerHandle& sampler);

  public:
    ~Texture();

    // -----------------------------------------------------------

    operator const vk::Image&() const {
        return *view->get_image();
    }

    operator const vk::ImageView&() const {
        return *view;
    }

    // Convenience method for view->get_image()
    const ImageHandle& get_image() const {
        return view->get_image();
    }

    const ImageViewHandle& get_view() const {
        return view;
    }

    const SamplerHandle& get_sampler() const {
        return sampler;
    }

    // Convenience method for get_image()->get_current_layout()
    const vk::ImageLayout& get_current_layout() const {
        return view->get_image()->get_current_layout();
    }

    vk::DescriptorImageInfo
    get_descriptor_info(const std::optional<vk::ImageLayout> access_layout = std::nullopt) const;

    // -----------------------------------------------------------

    void properties(Properties& props);

  private:
    const ImageViewHandle view;
    SamplerHandle sampler;

  public:
    static inline const TextureHandle EMPTY = nullptr;

    static TextureHandle create(const ImageViewHandle& view, const SamplerHandle& sampler);
};

class AccelerationStructure;
using AccelerationStructureHandle = std::shared_ptr<AccelerationStructure>;

class AccelerationStructure : public std::enable_shared_from_this<AccelerationStructure>,
                              public Resource {
  protected:
    // Creats a AccelerationStructure objects that automatically destroys `as` when destructed.
    // The memory is not freed explicitly to let it free itself.
    // It is asserted that the memory is already bound correctly.
    AccelerationStructure(const vk::AccelerationStructureKHR& as,
                          const BufferHandle& buffer,
                          const vk::AccelerationStructureBuildSizesInfoKHR& size_info);

  public:
    ~AccelerationStructure();

    // -----------------------------------------------------------

    operator const vk::AccelerationStructureKHR&() const {
        return as;
    }

    operator const vk::AccelerationStructureKHR*() const {
        return &as;
    }

    const vk::AccelerationStructureKHR& operator*() const {
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

    vk::WriteDescriptorSetAccelerationStructureKHR get_descriptor_info() const {
        return vk::WriteDescriptorSetAccelerationStructureKHR(1, &as);
    }

    // -----------------------------------------------------------

    // E.g. needed for accelerationStructureReference in VkAccelerationStructureInstanceKHR
    vk::DeviceAddress get_acceleration_structure_device_address() const;

    // A barrier to insert between tlas builds and tlas usage.
    vk::BufferMemoryBarrier2 tlas_read_barrier2(const vk::PipelineStageFlags2 read_stages) const;

    // A barier to insert between blas builds and blas usage.
    vk::BufferMemoryBarrier2 blas_read_barrier2() const;

    // A barier to insert between blas builds and blas usage.
    vk::BufferMemoryBarrier blas_read_barrier() const;

    // A barrier to insert between tlas usage and tlas rebuild/update.
    vk::BufferMemoryBarrier2 tlas_build_barrier2(const vk::PipelineStageFlags2 read_stages) const;

    // A barier to insert between blas read (for TLAS build) and blas rebuild/update.
    vk::BufferMemoryBarrier2 blas_build_barrier2() const;

    // A barier to insert between blas usage and blas rebuild/update.
    vk::BufferMemoryBarrier blas_build_barrier() const;

    // -----------------------------------------------------------

    void properties(Properties& props);

  private:
    const vk::AccelerationStructureKHR as;
    const BufferHandle buffer;
    const vk::AccelerationStructureBuildSizesInfoKHR size_info;

  public:
    static AccelerationStructureHandle
    create(const vk::AccelerationStructureKHR& as,
           const BufferHandle& buffer,
           const vk::AccelerationStructureBuildSizesInfoKHR& size_info);
};

} // namespace merian
