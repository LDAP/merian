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
    : buffer(buffer), memory(memory), create_info(create_info) {}

Buffer::~Buffer() {
    SPDLOG_TRACE("destroy buffer ({})", fmt::ptr(static_cast<VkBuffer>(buffer)));
    memory->get_context()->device.destroyBuffer(buffer);
}

vk::DeviceAddress Buffer::get_device_address() {
    assert(create_info.usage | vk::BufferUsageFlagBits::eShaderDeviceAddress);
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

BufferHandle Buffer::create_aliasing_buffer() {
    return get_memory()->create_aliasing_buffer(create_info);
}

void Buffer::properties(Properties& props) {
    props.output_text(fmt::format("Handle: {}\nSize: {}\nUsage: {}",
                                  fmt::ptr(static_cast<VkBuffer>(buffer)),
                                  format_size(create_info.size), vk::to_string(create_info.usage)));
    if (props.st_begin_child("memory_info", "Memory")) {
        get_memory()->properties(props);
        props.st_end_child();
    }
}

// --------------------------------------------------------------------------

Image::Image(const vk::Image& image,
             const MemoryAllocationHandle& memory,
             const vk::ImageCreateInfo create_info,
             const vk::ImageLayout current_layout)
    : image(image), memory(memory), create_info(create_info), current_layout(current_layout) {}

Image::~Image() {
    SPDLOG_TRACE("destroy image ({})", fmt::ptr(static_cast<VkImage>(image)));
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

vk::ImageMemoryBarrier Image::barrier(const vk::ImageLayout new_layout) {
    return barrier(new_layout, access_flags_for_image_layout(current_layout),
                   access_flags_for_image_layout(new_layout));
}

vk::ImageMemoryBarrier2 Image::barrier2(const vk::ImageLayout new_layout) {
    return barrier2(new_layout, access_flags2_for_image_layout(current_layout),
                    access_flags2_for_image_layout(new_layout),
                    pipeline_stage2_for_image_layout(current_layout),
                    pipeline_stage2_for_image_layout(new_layout));
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

    if (create_info.usage & VALID_IMAGE_USAGE_FOR_IMAGE_VIEWS) {
        return true;
    }
    return false;
}

ImageHandle Image::create_aliasing_image() {
    return get_memory()->create_aliasing_image(create_info);
}

void Image::properties(Properties& props) {
    props.output_text(fmt::format(
        "Handle: {}\nExtent: {} x {} x {}\nUsage: {}\nTiling: {}\nFormat: {}\nCurrent Layout: {}",
        fmt::ptr(static_cast<VkImage>(image)), get_extent().width, get_extent().height,
        get_extent().depth, vk::to_string(get_usage_flags()), vk::to_string(get_tiling()),
        vk::to_string(get_format()), vk::to_string(get_current_layout())));
    if (props.st_begin_child("memory_info", "Memory")) {
        get_memory()->properties(props);
        props.st_end_child();
    }
}

// --------------------------------------------------------------------------

Texture::Texture(const vk::ImageView& view, const ImageHandle& image, const SamplerHandle& sampler)
    : view(view), image(image), sampler(sampler) {
    assert(sampler);
}

Texture::~Texture() {
    SPDLOG_TRACE("destroy image view ({})", fmt::ptr(static_cast<VkImageView>(view)));
    image->get_memory()->get_context()->device.destroyImageView(view);
}

void Texture::set_sampler(const SamplerHandle& sampler) {
    assert(sampler);
    this->sampler = sampler;
}

void Texture::properties(Properties& props) {
    if (props.st_begin_child("image_info", "Image")) {
        image->properties(props);
        props.st_end_child();
    }
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
    if (props.st_begin_child("buffer_info", "Buffer")) {
        buffer->properties(props);
        props.st_end_child();
    }
}

} // namespace merian
