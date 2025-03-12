#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/utils/barriers.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

namespace merian {

Buffer::Buffer(const vk::Buffer& buffer,
               const MemoryAllocationHandle& memory,
               const vk::BufferCreateInfo& create_info)
    : context(memory->get_context()), buffer(buffer), memory(memory), create_info(create_info) {}

Buffer::Buffer(const ContextHandle& context, const vk::BufferCreateInfo& create_info)
    : context(context), buffer(context->device.createBuffer(create_info)),
      create_info(create_info) {}

Buffer::~Buffer() {
    SPDLOG_TRACE("destroy buffer ({})", fmt::ptr(static_cast<VkBuffer>(buffer)));
    context->device.destroyBuffer(buffer);
}

vk::MemoryRequirements Buffer::get_memory_requirements() const {
    return context->device.getBufferMemoryRequirements(buffer);
}

vk::DeviceAddress Buffer::get_device_address() const {
    assert(create_info.usage | vk::BufferUsageFlagBits::eShaderDeviceAddress);
    return context->device.getBufferAddress(get_buffer_device_address_info());
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

BufferHandle Buffer::create_aliasing_buffer() const {
    assert(memory && "buffer is not bound to memory, cannot create aliasing buffer.");
    return memory->create_aliasing_buffer(create_info);
}

void Buffer::properties(Properties& props) {
    props.output_text(fmt::format("Handle: {}\nSize: {}\nUsage: {}",
                                  fmt::ptr(static_cast<VkBuffer>(buffer)),
                                  format_size(create_info.size), vk::to_string(create_info.usage)));
    if (memory && props.st_begin_child("memory_info", "Memory")) {
        get_memory()->properties(props);
        props.st_end_child();
    }
}

BufferHandle Buffer::create(const vk::Buffer& buffer,
                            const MemoryAllocationHandle& memory,
                            const vk::BufferCreateInfo& create_info) {
    return std::shared_ptr<Buffer>(new Buffer(buffer, memory, create_info));
}

BufferHandle Buffer::create(const ContextHandle& context, const vk::BufferCreateInfo& create_info) {
    return std::shared_ptr<Buffer>(new Buffer(context, create_info));
}

// --------------------------------------------------------------------------

Image::Image(const vk::Image& image,
             const MemoryAllocationHandle& memory,
             const vk::ImageCreateInfo create_info,
             const vk::ImageLayout current_layout)
    : context(memory->get_context()), image(image), memory(memory), create_info(create_info),
      current_layout(current_layout) {}

Image::Image(const ContextHandle& context,
             const vk::Image& image,
             const vk::ImageCreateInfo create_info,
             const vk::ImageLayout current_layout)
    : context(context), image(image), create_info(create_info), current_layout(current_layout) {}

Image::Image(const ContextHandle& context, const vk::ImageCreateInfo create_info)
    : context(context), image(context->device.createImage(create_info)), create_info(create_info),
      current_layout(create_info.initialLayout) {}

Image::~Image() {
    SPDLOG_TRACE("destroy image ({})", fmt::ptr(static_cast<VkImage>(image)));
    context->device.destroyImage(image);
}

vk::MemoryRequirements Image::get_memory_requirements() const {
    return context->device.getImageMemoryRequirements(image);
}

vk::FormatFeatureFlags Image::format_features() const {
    if (get_tiling() == vk::ImageTiling::eOptimal) {
        return context->physical_device.physical_device.getFormatProperties(get_format())
            .optimalTilingFeatures;
    }
    return context->physical_device.physical_device.getFormatProperties(get_format())
        .linearTilingFeatures;
}

vk::ImageMemoryBarrier Image::barrier(const vk::ImageLayout new_layout,
                                      const bool transition_from_undefined) {
    return barrier(new_layout, access_flags_for_image_layout(current_layout),
                   access_flags_for_image_layout(new_layout), VK_QUEUE_FAMILY_IGNORED,
                   VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), transition_from_undefined);
}

vk::ImageMemoryBarrier2 Image::barrier2(const vk::ImageLayout new_layout,
                                        const bool transition_from_undefined) {
    return barrier2(new_layout, access_flags2_for_image_layout(current_layout),
                    access_flags2_for_image_layout(new_layout),
                    pipeline_stage2_for_image_layout(current_layout),
                    pipeline_stage2_for_image_layout(new_layout), VK_QUEUE_FAMILY_IGNORED,
                    VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), transition_from_undefined);
}

// Do not forget submit the barrier, else the internal state does not match the actual state
vk::ImageMemoryBarrier Image::barrier(const vk::ImageLayout new_layout,
                                      const vk::AccessFlags src_access_flags,
                                      const vk::AccessFlags dst_access_flags,
                                      const uint32_t src_queue_family_index,
                                      const uint32_t dst_queue_family_index,
                                      const vk::ImageSubresourceRange subresource_range,
                                      const bool transition_from_undefined) {
    const vk::ImageLayout old_layout =
        transition_from_undefined ? vk::ImageLayout::eUndefined : current_layout;
    const vk::ImageMemoryBarrier barrier{
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

    const vk::ImageLayout old_layout =
        transition_from_undefined ? vk::ImageLayout::eUndefined : current_layout;
    const vk::ImageMemoryBarrier2 barrier{
        src_stage_flags, src_access_flags, dst_stage_flags,        dst_access_flags,
        old_layout,      new_layout,       src_queue_family_index, dst_queue_family_index,
        image,           subresource_range};
    current_layout = new_layout;

    return barrier;
}

bool Image::valid_for_view() {
    static const vk::ImageUsageFlags VALID_IMAGE_USAGE_FOR_IMAGE_VIEWS =
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment |
        vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eInputAttachment |
        vk::ImageUsageFlagBits::eFragmentShadingRateAttachmentKHR |
        vk::ImageUsageFlagBits::eFragmentDensityMapEXT |
        vk::ImageUsageFlagBits::eVideoDecodeDstKHR | vk::ImageUsageFlagBits::eVideoDecodeDpbKHR |
#if defined(VK_ENABLE_BETA_EXTENSIONS)
        vk::ImageUsageFlagBits::eVideoEncodeSrcKHR | vk::ImageUsageFlagBits::eVideoEncodeDpbKHR |
#endif
        vk::ImageUsageFlagBits::eSampleWeightQCOM | vk::ImageUsageFlagBits::eSampleBlockMatchQCOM;

    return static_cast<bool>(create_info.usage & VALID_IMAGE_USAGE_FOR_IMAGE_VIEWS);
}

ImageHandle Image::create_aliasing_image() {
    assert(memory && "image is not bound to memory, cannot create aliasing image.");
    return memory->create_aliasing_image(create_info);
}

void Image::properties(Properties& props) {
    props.output_text(fmt::format(
        "Handle: {}\nExtent: {} x {} x {}\nUsage: {}\nTiling: {}\nFormat: {}\nCurrent Layout: {}",
        fmt::ptr(static_cast<VkImage>(image)), get_extent().width, get_extent().height,
        get_extent().depth, vk::to_string(get_usage_flags()), vk::to_string(get_tiling()),
        vk::to_string(get_format()), vk::to_string(get_current_layout())));
    if (memory && props.st_begin_child("memory_info", "Memory")) {
        get_memory()->properties(props);
        props.st_end_child();
    }
}

ImageHandle Image::create(const vk::Image& image,
                          const MemoryAllocationHandle& memory,
                          const vk::ImageCreateInfo create_info,
                          const vk::ImageLayout current_layout) {
    return std::shared_ptr<Image>(new Image(image, memory, create_info, current_layout));
}

ImageHandle Image::create(const ContextHandle& context,
                          const vk::Image& image,
                          const vk::ImageCreateInfo create_info,
                          const vk::ImageLayout current_layout) {
    return std::shared_ptr<Image>(new Image(context, image, create_info, current_layout));
}

ImageHandle Image::create(const ContextHandle& context, const vk::ImageCreateInfo create_info) {
    return std::shared_ptr<Image>(new Image(context, create_info));
}

vk::DeviceSize Image::format_size(const vk::Format format) {
    switch (format) {
    case vk::Format::eR4G4UnormPack8:
        return 1;
    case vk::Format::eR4G4B4A4UnormPack16:
    case vk::Format::eB4G4R4A4UnormPack16:
    case vk::Format::eR5G6B5UnormPack16:
    case vk::Format::eB5G6R5UnormPack16:
    case vk::Format::eR5G5B5A1UnormPack16:
    case vk::Format::eB5G5R5A1UnormPack16:
    case vk::Format::eA1R5G5B5UnormPack16:
        return 2;
    case vk::Format::eR8Unorm:
    case vk::Format::eR8Snorm:
    case vk::Format::eR8Uscaled:
    case vk::Format::eR8Sscaled:
    case vk::Format::eR8Uint:
    case vk::Format::eR8Sint:
    case vk::Format::eR8Srgb:
        return 1;
    case vk::Format::eR8G8Unorm:
    case vk::Format::eR8G8Snorm:
    case vk::Format::eR8G8Uscaled:
    case vk::Format::eR8G8Sscaled:
    case vk::Format::eR8G8Uint:
    case vk::Format::eR8G8Sint:
    case vk::Format::eR8G8Srgb:
        return 2;
    case vk::Format::eR8G8B8Unorm:
    case vk::Format::eR8G8B8Snorm:
    case vk::Format::eR8G8B8Uscaled:
    case vk::Format::eR8G8B8Sscaled:
    case vk::Format::eR8G8B8Uint:
    case vk::Format::eR8G8B8Sint:
    case vk::Format::eR8G8B8Srgb:
    case vk::Format::eB8G8R8Unorm:
    case vk::Format::eB8G8R8Snorm:
    case vk::Format::eB8G8R8Uscaled:
    case vk::Format::eB8G8R8Sscaled:
    case vk::Format::eB8G8R8Uint:
    case vk::Format::eB8G8R8Sint:
    case vk::Format::eB8G8R8Srgb:
        return 3;
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Snorm:
    case vk::Format::eR8G8B8A8Uscaled:
    case vk::Format::eR8G8B8A8Sscaled:
    case vk::Format::eR8G8B8A8Uint:
    case vk::Format::eR8G8B8A8Sint:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Snorm:
    case vk::Format::eB8G8R8A8Uscaled:
    case vk::Format::eB8G8R8A8Sscaled:
    case vk::Format::eB8G8R8A8Uint:
    case vk::Format::eB8G8R8A8Sint:
    case vk::Format::eB8G8R8A8Srgb:
        return 4;
    case vk::Format::eA8B8G8R8UnormPack32:
    case vk::Format::eA8B8G8R8SnormPack32:
    case vk::Format::eA8B8G8R8UscaledPack32:
    case vk::Format::eA8B8G8R8SscaledPack32:
    case vk::Format::eA8B8G8R8UintPack32:
    case vk::Format::eA8B8G8R8SintPack32:
    case vk::Format::eA8B8G8R8SrgbPack32:
    case vk::Format::eA2R10G10B10UnormPack32:
    case vk::Format::eA2R10G10B10SnormPack32:
    case vk::Format::eA2R10G10B10UscaledPack32:
    case vk::Format::eA2R10G10B10SscaledPack32:
    case vk::Format::eA2R10G10B10UintPack32:
    case vk::Format::eA2R10G10B10SintPack32:
    case vk::Format::eA2B10G10R10UnormPack32:
    case vk::Format::eA2B10G10R10SnormPack32:
    case vk::Format::eA2B10G10R10UscaledPack32:
    case vk::Format::eA2B10G10R10SscaledPack32:
    case vk::Format::eA2B10G10R10UintPack32:
    case vk::Format::eA2B10G10R10SintPack32:
        return 4;
    case vk::Format::eR16Unorm:
    case vk::Format::eR16Snorm:
    case vk::Format::eR16Uscaled:
    case vk::Format::eR16Sscaled:
    case vk::Format::eR16Uint:
    case vk::Format::eR16Sint:
    case vk::Format::eR16Sfloat:
        return 2;
    case vk::Format::eR16G16Unorm:
    case vk::Format::eR16G16Snorm:
    case vk::Format::eR16G16Uscaled:
    case vk::Format::eR16G16Sscaled:
    case vk::Format::eR16G16Uint:
    case vk::Format::eR16G16Sint:
    case vk::Format::eR16G16Sfloat:
        return 4;
    case vk::Format::eR16G16B16Unorm:
    case vk::Format::eR16G16B16Snorm:
    case vk::Format::eR16G16B16Uscaled:
    case vk::Format::eR16G16B16Sscaled:
    case vk::Format::eR16G16B16Uint:
    case vk::Format::eR16G16B16Sint:
    case vk::Format::eR16G16B16Sfloat:
        return 6;
    case vk::Format::eR16G16B16A16Unorm:
    case vk::Format::eR16G16B16A16Snorm:
    case vk::Format::eR16G16B16A16Uscaled:
    case vk::Format::eR16G16B16A16Sscaled:
    case vk::Format::eR16G16B16A16Uint:
    case vk::Format::eR16G16B16A16Sint:
    case vk::Format::eR16G16B16A16Sfloat:
        return 8;
    case vk::Format::eR32Uint:
    case vk::Format::eR32Sint:
    case vk::Format::eR32Sfloat:
        return 4;
    case vk::Format::eR32G32Uint:
    case vk::Format::eR32G32Sint:
    case vk::Format::eR32G32Sfloat:
        return 8;
    case vk::Format::eR32G32B32Uint:
    case vk::Format::eR32G32B32Sint:
    case vk::Format::eR32G32B32Sfloat:
        return 12;
    case vk::Format::eR32G32B32A32Uint:
    case vk::Format::eR32G32B32A32Sint:
    case vk::Format::eR32G32B32A32Sfloat:
        return 16;
    case vk::Format::eR64Uint:
    case vk::Format::eR64Sint:
    case vk::Format::eR64Sfloat:
        return 8;
    case vk::Format::eR64G64Uint:
    case vk::Format::eR64G64Sint:
    case vk::Format::eR64G64Sfloat:
        return 16;
    case vk::Format::eR64G64B64Uint:
    case vk::Format::eR64G64B64Sint:
    case vk::Format::eR64G64B64Sfloat:
        return 24;
    case vk::Format::eR64G64B64A64Uint:
    case vk::Format::eR64G64B64A64Sint:
    case vk::Format::eR64G64B64A64Sfloat:
        return 32;
    case vk::Format::eB10G11R11UfloatPack32:
    case vk::Format::eE5B9G9R9UfloatPack32:
        return 4;
    case vk::Format::eD16Unorm:
        return 2;
    case vk::Format::eX8D24UnormPack32:
        return 4;
    case vk::Format::eD32Sfloat:
        return 4;
    case vk::Format::eS8Uint:
        return 1;
    case vk::Format::eD16UnormS8Uint:
        return 3;
    case vk::Format::eD24UnormS8Uint:
        return 4;
    case vk::Format::eD32SfloatS8Uint:
        return 5;
    default:
        throw std::runtime_error{"unsupported"};
    }
}

// --------------------------------------------------------------------------

ImageView::ImageView(const vk::ImageViewCreateInfo& view_create_info, const ImageHandle& image)
    : image(image) {
    assert(image->valid_for_view());
    assert(view_create_info.image == **image);

    view = image->get_context()->device.createImageView(view_create_info);
    SPDLOG_TRACE("create image view ({})", fmt::ptr(static_cast<VkImageView>(view)));
}

ImageView::ImageView(const vk::ImageView& view, const ImageHandle& image)
    : view(view), image(image) {
    SPDLOG_TRACE("create image view ({})", fmt::ptr(static_cast<VkImageView>(view)));
}

ImageView::~ImageView() {
    SPDLOG_TRACE("destroy image view ({})", fmt::ptr(static_cast<VkImageView>(view)));
    image->get_context()->device.destroyImageView(view);
}

void ImageView::properties(Properties& props) {
    if (props.st_begin_child("image_info", "Image")) {
        image->properties(props);
        props.st_end_child();
    }
}

ImageViewHandle ImageView::create(const vk::ImageViewCreateInfo& view_create_info,
                                  const ImageHandle& image) {
    return std::shared_ptr<ImageView>(new ImageView(view_create_info, image));
}

ImageViewHandle ImageView::create(const vk::ImageView& view, const ImageHandle& image) {
    return std::shared_ptr<ImageView>(new ImageView(view, image));
}

// --------------------------------------------------------------------------

Texture::Texture(const ImageViewHandle& view, const SamplerHandle& sampler)
    : view(view), sampler(sampler) {
    assert(view);
    assert(sampler);
}

Texture::~Texture() {}

void Texture::set_sampler(const SamplerHandle& sampler) {
    assert(sampler);
    this->sampler = sampler;
}

void Texture::properties(Properties& props) {
    if (props.st_begin_child("image_view_info", "Image View")) {
        view->properties(props);
        props.st_end_child();
    }
}

TextureHandle Texture::create(const ImageViewHandle& view, const SamplerHandle& sampler) {
    return std::shared_ptr<Texture>(new Texture(view, sampler));
}

// --------------------------------------------------------------------------

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

void AccelerationStructure::properties(Properties& props) {
    props.output_text(fmt::format("Size: {}\nBuild scratch size: {}\nUpdate scratch size: {}",
                                  format_size(size_info.accelerationStructureSize),
                                  format_size(size_info.buildScratchSize),
                                  format_size(size_info.updateScratchSize)));
    if (props.st_begin_child("buffer_info", "Buffer")) {
        buffer->properties(props);
        props.st_end_child();
    }
}

AccelerationStructureHandle
AccelerationStructure::create(const vk::AccelerationStructureKHR& as,
                              const BufferHandle& buffer,
                              const vk::AccelerationStructureBuildSizesInfoKHR& size_info) {
    return std::shared_ptr<AccelerationStructure>(new AccelerationStructure(as, buffer, size_info));
}

} // namespace merian
