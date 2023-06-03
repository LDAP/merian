#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/barriers.hpp"
#include "merian/vk/utils/check_result.hpp"

namespace merian {

ResourceAllocator::ResourceAllocator(vk::Device device,
                                     vk::PhysicalDevice physicalDevice,
                                     MemoryAllocator* memAllocator,
                                     SamplerPool& samplerPool,
                                     vk::DeviceSize stagingBlockSize)
    : m_samplerPool(samplerPool) {
    init(device, physicalDevice, memAllocator, stagingBlockSize);
}

ResourceAllocator::~ResourceAllocator() {
    deinit();
}

void ResourceAllocator::init(vk::Device device,
                             vk::PhysicalDevice physicalDevice,
                             MemoryAllocator* memAlloc,
                             vk::DeviceSize stagingBlockSize) {
    m_device = device;
    m_physicalDevice = physicalDevice;
    m_memAlloc = memAlloc;
    m_memoryProperties = physicalDevice.getMemoryProperties();
    m_staging = std::make_unique<StagingMemoryManager>(memAlloc, stagingBlockSize);
}

void ResourceAllocator::deinit() {
    m_staging.reset();
}

Buffer ResourceAllocator::createBuffer(const vk::BufferCreateInfo& info_,
                                       const vk::MemoryPropertyFlags memProperties_,
                                       const vk::DeviceSize alignment) {
    Buffer resultBuffer;
    // Create Buffer (can be overloaded)
    CreateBufferEx(info_, &resultBuffer.buffer);
    resultBuffer.usage = info_.usage;

    // Find memory requirements
    vk::MemoryRequirements2 memReqs;
    vk::MemoryDedicatedRequirements dedicatedRegs;
    vk::BufferMemoryRequirementsInfo2 bufferReqs;

    memReqs.pNext = &dedicatedRegs;
    bufferReqs.buffer = resultBuffer.buffer;
    m_device.getBufferMemoryRequirements2(&bufferReqs, &memReqs);

    // Scratch Buffer needs extra alignment
    memReqs.memoryRequirements.alignment =
        std::max(memReqs.memoryRequirements.alignment, alignment);

    // Build up allocation info
    MemAllocateInfo allocInfo(memReqs.memoryRequirements, memProperties_, false);

    if (info_.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        allocInfo.setAllocationFlags(vk::MemoryAllocateFlagBits::eDeviceAddress);
    }
    if (dedicatedRegs.requiresDedicatedAllocation) {
        allocInfo.setDedicatedBuffer(resultBuffer.buffer);
    }

    // Allocate memory
    resultBuffer.memHandle = AllocateMemory(allocInfo);
    if (resultBuffer.memHandle) {
        const auto memInfo = m_memAlloc->getMemoryInfo(resultBuffer.memHandle);
        m_device.bindBufferMemory(resultBuffer.buffer, memInfo.memory, memInfo.offset);
    } else {
        destroy(resultBuffer);
    }

    return resultBuffer;
}

Buffer ResourceAllocator::createBuffer(vk::DeviceSize size_,
                                       vk::BufferUsageFlags usage_,
                                       const vk::MemoryPropertyFlags memUsage_,
                                       const vk::DeviceSize alignment) {
    vk::BufferCreateInfo info{{}, size_, vk::BufferUsageFlagBits::eTransferDst | usage_};
    return createBuffer(info, memUsage_, alignment);
}

Buffer ResourceAllocator::createBuffer(const vk::CommandBuffer& cmdBuf,
                                       const vk::DeviceSize& size_,
                                       const void* data_,
                                       vk::BufferUsageFlags usage_,
                                       vk::MemoryPropertyFlags memProps) {
    vk::BufferCreateInfo createInfoR{{}, size_, usage_ | vk::BufferUsageFlagBits::eTransferDst};
    Buffer resultBuffer = createBuffer(createInfoR, memProps);

    if (data_) {
        m_staging->cmdToBuffer(cmdBuf, resultBuffer.buffer, 0, size_, data_);
    }

    return resultBuffer;
}

// You get the alignment from
// VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment
Buffer ResourceAllocator::createScratchBuffer(const vk::DeviceSize size,
                                              const vk::DeviceSize alignment) {
    return createBuffer(size,
                        vk::BufferUsageFlagBits::eShaderDeviceAddress |
                            vk::BufferUsageFlagBits::eStorageBuffer,
                        vk::MemoryPropertyFlagBits::eDeviceLocal, alignment);
}

Image ResourceAllocator::createImage(const vk::ImageCreateInfo& info_,
                                     const vk::MemoryPropertyFlags memUsage_) {
    Image resultImage;
    // Create image
    CreateImageEx(info_, &resultImage.image);

    // Find memory requirements
    vk::MemoryRequirements2 memReqs;
    vk::MemoryDedicatedRequirements dedicatedRegs;
    vk::ImageMemoryRequirementsInfo2 imageReqs;

    imageReqs.image = resultImage.image;
    memReqs.pNext = &dedicatedRegs;

    m_device.getImageMemoryRequirements2(&imageReqs, &memReqs);

    // Build up allocation info
    MemAllocateInfo allocInfo(memReqs.memoryRequirements, memUsage_, true);
    if (dedicatedRegs.requiresDedicatedAllocation) {
        allocInfo.setDedicatedImage(resultImage.image);
    }

    // Allocate memory
    resultImage.memHandle = AllocateMemory(allocInfo);
    if (resultImage.memHandle) {
        const auto memInfo = m_memAlloc->getMemoryInfo(resultImage.memHandle);
        m_device.bindImageMemory(resultImage.image, memInfo.memory, memInfo.offset);
    } else {
        destroy(resultImage);
    }
    return resultImage;
}

Image ResourceAllocator::createImage(const vk::CommandBuffer& cmdBuf,
                                     size_t size_,
                                     const void* data_,
                                     const vk::ImageCreateInfo& info_,
                                     const vk::ImageLayout& layout_) {
    Image resultImage = createImage(info_, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Copy the data to staging buffer than to image
    if (data_ != nullptr) {
        // Copy buffer to image
        vk::ImageSubresourceRange subresourceRange{vk::ImageAspectFlagBits::eColor, 0,
                                                   info_.mipLevels, 0, 1};

        // doing these transitions per copy is not efficient, should do in bulk for many images
        merian::cmd_barrier_image_layout(cmdBuf, resultImage.image, vk::ImageLayout::eUndefined,
                                      vk::ImageLayout::eTransferDstOptimal, subresourceRange);

        vk::Offset3D offset;
        vk::ImageSubresourceLayers subresource{vk::ImageAspectFlagBits::eColor, {}, {}, 1};
        m_staging->cmdToImage(cmdBuf, resultImage.image, offset, info_.extent, subresource, size_,
                              data_);

        // Setting final image layout
        merian::cmd_barrier_image_layout(cmdBuf, resultImage.image,
                                      vk::ImageLayout::eTransferDstOptimal, layout_);
    } else {
        // Setting final image layout
        merian::cmd_barrier_image_layout(cmdBuf, resultImage.image, vk::ImageLayout::eUndefined,
                                      layout_);
    }

    return resultImage;
}

merian::Texture ResourceAllocator::createTexture(const Image& image,
                                                 const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                                 const vk::SamplerCreateInfo& samplerCreateInfo) {
    Texture resultTexture = createTexture(image, imageViewCreateInfo);
    resultTexture.descriptor.sampler = m_samplerPool.acquireSampler(samplerCreateInfo);

    return resultTexture;
}

Texture ResourceAllocator::createTexture(const Image& image,
                                         const vk::ImageViewCreateInfo& imageViewCreateInfo) {
    assert(imageViewCreateInfo.image == image.image);

    Texture resultTexture;

    resultTexture.image = image.image;
    resultTexture.memHandle = image.memHandle;
    resultTexture.descriptor.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    resultTexture.descriptor.imageView = m_device.createImageView(imageViewCreateInfo);

    return resultTexture;
}

Texture ResourceAllocator::createTexture(const vk::CommandBuffer& cmdBuf,
                                         size_t size_,
                                         const void* data_,
                                         const vk::ImageCreateInfo& info_,
                                         const vk::SamplerCreateInfo& samplerCreateInfo,
                                         const vk::ImageLayout& layout_,
                                         bool isCube) {
    Image image = createImage(cmdBuf, size_, data_, info_, layout_);

    vk::ImageSubresourceRange subresourceRange{
        vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS,
    };
    vk::ImageViewCreateInfo viewInfo{
        {}, image.image, {}, info_.format, {}, subresourceRange,
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

    Texture resultTexture = createTexture(image, viewInfo, samplerCreateInfo);
    resultTexture.descriptor.imageLayout = layout_;
    return resultTexture;
}

AccelerationStructure
ResourceAllocator::createAccelerationStructure(vk::DeviceSize size,
                                               vk::AccelerationStructureTypeKHR type) {
    AccelerationStructure resultAccel;
    // Allocating the buffer to hold the acceleration structure
    resultAccel.buffer =
        createBuffer(size, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                               vk::BufferUsageFlagBits::eShaderDeviceAddress);
    // Setting the buffer
    vk::AccelerationStructureCreateInfoKHR createInfo{
        {}, resultAccel.buffer.buffer, {}, size, type};
    check_result(m_device.createAccelerationStructureKHR(&createInfo, nullptr, &resultAccel.as),
                 "could not create acceleration structure");

    return resultAccel;
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

merian::StagingMemoryManager* ResourceAllocator::getStaging() {
    return m_staging.get();
}

const merian::StagingMemoryManager* ResourceAllocator::getStaging() const {
    return m_staging.get();
}

void ResourceAllocator::destroy(Buffer& b_) {
    m_device.destroyBuffer(b_.buffer, nullptr);
    m_memAlloc->freeMemory(b_.memHandle);

    b_ = Buffer();
}

void ResourceAllocator::destroy(Image& i_) {
    m_device.destroyImage(i_.image, nullptr);
    m_memAlloc->freeMemory(i_.memHandle);

    i_ = Image();
}

void ResourceAllocator::destroy(Texture& t_) {
    m_device.destroyImageView(t_.descriptor.imageView, nullptr);
    m_device.destroyImage(t_.image, nullptr);
    m_memAlloc->freeMemory(t_.memHandle);

    if (t_.descriptor.sampler) {
        m_samplerPool.releaseSampler(t_.descriptor.sampler);
    }

    t_ = Texture();
}

void ResourceAllocator::destroy(AccelerationStructure& a_) {
    m_device.destroyAccelerationStructureKHR(a_.as, nullptr);
    destroy(a_.buffer);

    a_ = AccelerationStructure();
}

void* ResourceAllocator::map(const Buffer& buffer) {
    void* pData = m_memAlloc->map(buffer.memHandle);
    return pData;
}

void ResourceAllocator::unmap(const Buffer& buffer) {
    m_memAlloc->unmap(buffer.memHandle);
}

void* ResourceAllocator::map(const Image& buffer) {
    void* pData = m_memAlloc->map(buffer.memHandle);
    return pData;
}

void ResourceAllocator::unmap(const Image& image) {
    m_memAlloc->unmap(image.memHandle);
}

MemHandle ResourceAllocator::AllocateMemory(const MemAllocateInfo& allocateInfo) {
    return m_memAlloc->allocMemory(allocateInfo);
}

void ResourceAllocator::CreateBufferEx(const vk::BufferCreateInfo& info_, vk::Buffer* buffer) {
    check_result(m_device.createBuffer(&info_, nullptr, buffer), "could not create buffer");
}

void ResourceAllocator::CreateImageEx(const vk::ImageCreateInfo& info_, vk::Image* image) {
    check_result(m_device.createImage(&info_, nullptr, image), "could not create image");
}

uint32_t ResourceAllocator::getMemoryType(uint32_t typeBits,
                                          const vk::MemoryPropertyFlags& properties) {
    for (uint32_t i = 0; i < m_memoryProperties.memoryTypeCount; i++) {
        if (((typeBits & (1 << i)) > 0) &&
            (m_memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error{"could not determine memory type"};
}

} // namespace merian
