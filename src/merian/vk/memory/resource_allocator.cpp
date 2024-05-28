#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/barriers.hpp"
#include "merian/vk/utils/check_result.hpp"
#include <spdlog/spdlog.h>

namespace merian {

ResourceAllocator::ResourceAllocator(const SharedContext& context,
                                     const std::shared_ptr<MemoryAllocator>& memAllocator,
                                     const std::shared_ptr<StagingMemoryManager> staging,
                                     const std::shared_ptr<SamplerPool>& samplerPool)
    : context(context), m_memAlloc(memAllocator), m_staging(staging), m_samplerPool(samplerPool) {
    SPDLOG_DEBUG("create ResourceAllocator ({})", fmt::ptr(this));
}

BufferHandle ResourceAllocator::createBuffer(const vk::BufferCreateInfo& info,
                                             const MemoryMappingType mapping_type,
                                             const std::string& debug_name,
                                             const std::optional<vk::DeviceSize> min_alignment) {
    return m_memAlloc->create_buffer(info, mapping_type, debug_name, min_alignment);
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

// You get the alignment from
// VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment
BufferHandle ResourceAllocator::createScratchBuffer(const vk::DeviceSize size,
                                                    const vk::DeviceSize alignment,
                                                    const std::string& debug_name) {
    return createBuffer(size,
                        vk::BufferUsageFlagBits::eShaderDeviceAddress |
                            vk::BufferUsageFlagBits::eStorageBuffer,
                        NONE, debug_name, alignment);
}

ImageHandle ResourceAllocator::createImage(const vk::ImageCreateInfo& info_,
                                           const MemoryMappingType mapping_type,
                                           const std::string& debug_name) {
    return m_memAlloc->create_image(info_, mapping_type, debug_name);
}

ImageHandle ResourceAllocator::createImage(const vk::CommandBuffer& cmdBuf,
                                           const size_t size_,
                                           const void* data_,
                                           const vk::ImageCreateInfo& info_,
                                           const MemoryMappingType mapping_type,
                                           const vk::ImageLayout& layout_,
                                           const std::string& debug_name) {
    ImageHandle resultImage = createImage(info_, mapping_type, debug_name);

    // Copy the data to staging buffer than to image
    if (data_ != nullptr) {
        // Copy buffer to image
        vk::ImageSubresourceRange subresourceRange{vk::ImageAspectFlagBits::eColor, 0,
                                                   info_.mipLevels, 0, 1};

        // doing these transitions per copy is not efficient, should do in bulk for many images
        merian::cmd_barrier_image_layout(cmdBuf, *resultImage, vk::ImageLayout::eUndefined,
                                         vk::ImageLayout::eTransferDstOptimal, subresourceRange);

        vk::Offset3D offset;
        vk::ImageSubresourceLayers subresource{vk::ImageAspectFlagBits::eColor, {}, {}, 1};
        m_staging->cmdToImage(cmdBuf, *resultImage, offset, info_.extent, subresource, size_,
                              data_);

        // Setting final image layout
        merian::cmd_barrier_image_layout(cmdBuf, *resultImage, vk::ImageLayout::eTransferDstOptimal,
                                         layout_);
        resultImage->_set_current_layout(layout_);
    } else {
        // Setting final image layout
        merian::cmd_barrier_image_layout(cmdBuf, *resultImage, vk::ImageLayout::eUndefined,
                                         layout_);
    }

    return resultImage;
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                               const SamplerHandle& sampler) {
    assert(imageViewCreateInfo.image == image->get_image());
    return std::make_shared<Texture>(image, imageViewCreateInfo, sampler);
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                               const vk::SamplerCreateInfo& samplerCreateInfo) {
    const SamplerHandle sampler = m_samplerPool->acquire_sampler(samplerCreateInfo);
    return createTexture(image, imageViewCreateInfo, sampler);
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image) {
    return createTexture(image, image->make_view_create_info());
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& view_create_info) {
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

    return createTexture(image, view_create_info, sampler);
}

TextureHandle ResourceAllocator::createTexture(const vk::CommandBuffer& cmdBuf,
                                               const size_t size_,
                                               const void* data_,
                                               const vk::ImageCreateInfo& info_,
                                               const MemoryMappingType mapping_type,
                                               const vk::SamplerCreateInfo& samplerCreateInfo,
                                               const vk::ImageLayout& layout_,
                                               const bool is_cube,
                                               const std::string& debug_name) {
    ImageHandle image = createImage(cmdBuf, size_, data_, info_, mapping_type, layout_, debug_name);
    return createTexture(image, image->make_view_create_info(is_cube), samplerCreateInfo);
}

AccelerationStructureHandle ResourceAllocator::createAccelerationStructure(
    const vk::AccelerationStructureTypeKHR type,
    const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
    const std::string& debug_name) {
    // Allocating the buffer to hold the acceleration structure
    BufferHandle buffer = createBuffer(size_info.accelerationStructureSize,
                                       vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                           vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                       NONE, debug_name);
    vk::AccelerationStructureKHR as;
    // Setting the buffer
    vk::AccelerationStructureCreateInfoKHR createInfo{
        {}, *buffer, {}, size_info.accelerationStructureSize, type};
    check_result(context->device.createAccelerationStructureKHR(&createInfo, nullptr, &as),
                 "could not create acceleration structure");

    return std::make_shared<AccelerationStructure>(as, buffer, size_info);
}

std::shared_ptr<StagingMemoryManager> ResourceAllocator::getStaging() {
    return m_staging;
}

const std::shared_ptr<StagingMemoryManager>& ResourceAllocator::getStaging() const {
    return m_staging;
}

} // namespace merian
