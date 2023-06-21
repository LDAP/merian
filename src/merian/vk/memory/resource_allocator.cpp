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
    } else {
        // Setting final image layout
        merian::cmd_barrier_image_layout(cmdBuf, *resultImage, vk::ImageLayout::eUndefined,
                                         layout_);
    }

    return resultImage;
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                               const vk::SamplerCreateInfo& samplerCreateInfo) {
    TextureHandle texture = createTexture(image, imageViewCreateInfo);
    texture->attach_sampler(m_samplerPool->acquire_sampler(samplerCreateInfo));
    return texture;
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& imageViewCreateInfo) {
    assert(imageViewCreateInfo.image == image->get_image());
    TextureHandle texture = std::make_shared<Texture>(image, imageViewCreateInfo);

    return texture;
}

TextureHandle ResourceAllocator::createTexture(const vk::CommandBuffer& cmdBuf,
                                               const size_t size_,
                                               const void* data_,
                                               const vk::ImageCreateInfo& info_,
                                               const MemoryMappingType mapping_type,
                                               const vk::SamplerCreateInfo& samplerCreateInfo,
                                               const vk::ImageLayout& layout_,
                                               const bool isCube,
                                               const std::string& debug_name) {
    ImageHandle image = createImage(cmdBuf, size_, data_, info_, mapping_type, layout_, debug_name);

    vk::ImageSubresourceRange subresourceRange{
        vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS,
    };
    vk::ImageViewCreateInfo viewInfo{
        {}, *image, {}, info_.format, {}, subresourceRange,
    };

    switch (info_.imageType) {
    case vk::ImageType::e1D:
        viewInfo.viewType =
            (info_.arrayLayers > 1 ? vk::ImageViewType::e1DArray : vk::ImageViewType::e1D);
        break;
    case vk::ImageType::e2D:
        if (isCube) {
            viewInfo.viewType = vk::ImageViewType::eCube;
        } else {
            viewInfo.viewType =
                info_.arrayLayers > 1 ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D;
        }
        break;
    case vk::ImageType::e3D:
        viewInfo.viewType = vk::ImageViewType::e3D;
        break;
    default:
        assert(0);
    }

    TextureHandle resultTexture = createTexture(image, viewInfo, samplerCreateInfo);
    return resultTexture;
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

void ResourceAllocator::finalizeStaging(vk::Fence fence) {
    m_staging->finalizeResources(fence);
}

void ResourceAllocator::releaseStaging() {
    m_staging->releaseResources();
}

void ResourceAllocator::finalizeAndReleaseStaging(vk::Fence fence) {
    m_staging->finalizeResources(fence);
    m_staging->releaseResources();
}

std::shared_ptr<StagingMemoryManager> ResourceAllocator::getStaging() {
    return m_staging;
}

const std::shared_ptr<StagingMemoryManager> ResourceAllocator::getStaging() const {
    return m_staging;
}

} // namespace merian
