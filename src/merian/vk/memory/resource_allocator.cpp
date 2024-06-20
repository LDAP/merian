#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/utils/colors.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/utils/check_result.hpp"
#include <spdlog/spdlog.h>

namespace merian {

ResourceAllocator::ResourceAllocator(const SharedContext& context,
                                     const std::shared_ptr<MemoryAllocator>& memAllocator,
                                     const std::shared_ptr<StagingMemoryManager> staging,
                                     const std::shared_ptr<SamplerPool>& samplerPool)
    : context(context), m_memAlloc(memAllocator), m_staging(staging), m_samplerPool(samplerPool),
      debug_utils(context->get_extension<ExtensionVkDebugUtils>()) {
    SPDLOG_DEBUG("create ResourceAllocator ({})", fmt::ptr(this));

    const uint32_t missing_rgba = merian::uint32_from_rgba(1, 0, 1, 1);
    const std::vector<uint32_t> data = {missing_rgba, missing_rgba, missing_rgba, missing_rgba};
    context->get_queue_GCT()->submit_wait([&](const vk::CommandBuffer& cmd) {
        dummy_texture = createTextureFromRGBA8(cmd, data.data(), 2, 2, vk::Filter::eLinear, true,
                                               "ResourceAllocator::dummy_texture");
        const auto img_transition =
            dummy_texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal);
        cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, img_transition});
        dummy_buffer = createBuffer(cmd, data.size() * sizeof(uint32_t),
                                    vk::BufferUsageFlagBits::eStorageBuffer, data.data(),
                                    MemoryMappingType::NONE, "ResourceAllocator::dummy_buffer");
    });

    SPDLOG_DEBUG("Uploaded dummy texture and buffer");
}

BufferHandle ResourceAllocator::createBuffer(const vk::BufferCreateInfo& info,
                                             const MemoryMappingType mapping_type,
                                             const std::string& debug_name,
                                             const std::optional<vk::DeviceSize> min_alignment) {
    const BufferHandle buffer =
        m_memAlloc->create_buffer(info, mapping_type, debug_name, min_alignment);

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->device, **buffer, debug_name);
    }
    SPDLOG_TRACE("created buffer {} ({})", fmt::ptr(static_cast<VkBuffer>(**buffer)), debug_name);
#endif

    return buffer;
}

BufferHandle ResourceAllocator::createBuffer(const vk::DeviceSize size_,
                                             const vk::BufferUsageFlags usage_,
                                             const MemoryMappingType mapping_type,
                                             const std::string& debug_name,
                                             const std::optional<vk::DeviceSize> min_alignment) {
    vk::BufferCreateInfo info{{}, size_, vk::BufferUsageFlagBits::eTransferDst | usage_};
    return createBuffer(info, mapping_type, debug_name, min_alignment);
}

BufferHandle ResourceAllocator::createBuffer(const vk::CommandBuffer& cmdBuf,
                                             const vk::DeviceSize& size_,
                                             const vk::BufferUsageFlags usage_,
                                             const void* data_,
                                             const MemoryMappingType mapping_type,
                                             const std::string& debug_name,
                                             const std::optional<vk::DeviceSize> min_alignment) {
    BufferHandle resultBuffer =
        createBuffer(size_, usage_, mapping_type, debug_name, min_alignment);

    if (data_) {
        m_staging->cmdToBuffer(cmdBuf, *resultBuffer, 0, size_, data_);
    }

    return resultBuffer;
}

const BufferHandle& ResourceAllocator::get_dummy_buffer() const {
    return dummy_buffer;
}

// You get the alignment from
// VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment
BufferHandle ResourceAllocator::createScratchBuffer(const vk::DeviceSize size,
                                                    const vk::DeviceSize alignment,
                                                    const std::string& debug_name) {
    return createBuffer(size,
                        vk::BufferUsageFlagBits::eShaderDeviceAddress |
                            vk::BufferUsageFlagBits::eStorageBuffer,
                        MemoryMappingType::NONE, debug_name, alignment);
}

BufferHandle ResourceAllocator::createInstancesBuffer(const uint32_t instance_count,
                                                      const std::string& debug_name) {
    static const vk::BufferUsageFlags instances_buffer_usage =
        vk::BufferUsageFlagBits::eShaderDeviceAddress |
        vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;

    return createBuffer(sizeof(vk::AccelerationStructureInstanceKHR) * instance_count,
                        instances_buffer_usage, merian::MemoryMappingType::NONE, debug_name, 16);
}

ImageHandle ResourceAllocator::createImage(const vk::ImageCreateInfo& info_,
                                           const MemoryMappingType mapping_type,
                                           const std::string& debug_name) {
    const ImageHandle image = m_memAlloc->create_image(info_, mapping_type, debug_name);

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->device, **image, debug_name);
    }
    SPDLOG_TRACE("created image {} ({})", fmt::ptr(static_cast<VkImage>(**image)), debug_name);
#endif

    return image;
}

ImageHandle ResourceAllocator::createImage(const vk::CommandBuffer& cmdBuf,
                                           const size_t size_,
                                           const void* data_,
                                           const vk::ImageCreateInfo& info_,
                                           const MemoryMappingType mapping_type,
                                           const std::string& debug_name) {
    assert(data_);

    const ImageHandle resultImage = createImage(info_, mapping_type, debug_name);
    // Copy buffer to image
    vk::ImageSubresourceRange subresourceRange{vk::ImageAspectFlagBits::eColor, 0, info_.mipLevels,
                                               0, 1};

    // doing these transitions per copy is not efficient, should do in bulk for many images
    cmdBuf.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
        resultImage->barrier(vk::ImageLayout::eTransferDstOptimal, {},
                             vk::AccessFlagBits::eTransferWrite, VK_QUEUE_FAMILY_IGNORED,
                             VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), true));

    m_staging->cmdToImage(cmdBuf, *resultImage, {}, info_.extent, first_layer(), size_, data_);

    return resultImage;
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& view_create_info,
                                               const SamplerHandle& sampler,
                                               [[maybe_unused]] const std::string& debug_name) {
    assert(view_create_info.image == image->get_image());

    const vk::ImageView view =
        image->get_memory()->get_context()->device.createImageView(view_create_info);
    const TextureHandle tex = std::make_shared<Texture>(view, image, sampler);

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->device, view, debug_name);
    }
    SPDLOG_TRACE("created image view {} ({}), for image {}",
                 fmt::ptr(static_cast<VkImageView>(view)), debug_name,
                 fmt::ptr(static_cast<VkImage>(**image)));
#endif

    return tex;
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                               const vk::SamplerCreateInfo& samplerCreateInfo,
                                               const std::string& debug_name) {
    const SamplerHandle sampler = m_samplerPool->acquire_sampler(samplerCreateInfo);
    return createTexture(image, imageViewCreateInfo, sampler, debug_name);
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const std::string& debug_name) {
    return createTexture(image, image->make_view_create_info(), debug_name);
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& view_create_info,
                                               const std::string& debug_name) {
    const SharedContext& context = image->get_memory()->get_context();

    const vk::FormatProperties props =
        context->physical_device.physical_device.getFormatProperties(view_create_info.format);

    SamplerHandle sampler;
    if ((image->get_tiling() == vk::ImageTiling::eOptimal &&
         (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) ||
        (image->get_tiling() == vk::ImageTiling::eLinear &&
         (props.linearTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))) {
        sampler = get_sampler_pool()->linear_mirrored_repeat();
    } else {
        sampler = get_sampler_pool()->nearest_mirrored_repeat();
    }

    return createTexture(image, view_create_info, sampler, debug_name);
}

TextureHandle ResourceAllocator::createTextureFromRGBA8(const vk::CommandBuffer& cmd,
                                                        const uint32_t* data,
                                                        const uint32_t width,
                                                        const uint32_t height,
                                                        const vk::Filter filter,
                                                        const bool isSRGB,
                                                        const std::string& debug_name) {
    const vk::ImageCreateInfo tex_image_info{
        {},
        vk::ImageType::e2D,
        isSRGB ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
        {width, height, 1},
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };
    const merian::ImageHandle image =
        createImage(cmd, width * height * sizeof(uint32_t), data, tex_image_info,
                    MemoryMappingType::NONE, debug_name);

    merian::SamplerHandle sampler =
        get_sampler_pool()->for_filter_and_address_mode(filter, vk::SamplerAddressMode::eRepeat);

    return createTexture(image, image->make_view_create_info(), sampler, debug_name);
}

const TextureHandle& ResourceAllocator::get_dummy_texture() const {
    return dummy_texture;
}

AccelerationStructureHandle ResourceAllocator::createAccelerationStructure(
    const vk::AccelerationStructureTypeKHR type,
    const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
    const std::string& debug_name) {
    // Allocating the buffer to hold the acceleration structure
    BufferHandle buffer = createBuffer(size_info.accelerationStructureSize,
                                       vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                           vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                       MemoryMappingType::NONE, debug_name);
    vk::AccelerationStructureKHR as;
    // Setting the buffer
    vk::AccelerationStructureCreateInfoKHR createInfo{
        {}, *buffer, {}, size_info.accelerationStructureSize, type};
    check_result(context->device.createAccelerationStructureKHR(&createInfo, nullptr, &as),
                 "could not create acceleration structure");

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->device, **buffer, debug_name);
        debug_utils->set_object_name(context->device, as, debug_name);
    }
#endif

    return std::make_shared<AccelerationStructure>(as, buffer, size_info);
}

std::shared_ptr<StagingMemoryManager> ResourceAllocator::getStaging() {
    return m_staging;
}

const std::shared_ptr<StagingMemoryManager>& ResourceAllocator::getStaging() const {
    return m_staging;
}

} // namespace merian
