#pragma once

#include "vk/memory/memory_allocator.hpp"
#include "vk/memory/resource_allocations.hpp"
#include "vk/memory/staging_memory_manager.hpp"
#include "vk/sampler/sampler_pool.hpp"

#include <memory>
#include <vulkan/vulkan.hpp>

namespace merian {
class ResourceAllocator {
  public:
    ResourceAllocator(ResourceAllocator const&) = delete;
    ResourceAllocator& operator=(ResourceAllocator const&) = delete;
    ResourceAllocator() = delete;

    ResourceAllocator(vk::Device device, vk::PhysicalDevice physicalDevice, MemoryAllocator* memAllocator,
                      SamplerPool& m_samplerPool, vk::DeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE);

    // All staging buffers must be cleared before
    virtual ~ResourceAllocator();

    //--------------------------------------------------------------------------------------------------
    // Initialization of the allocator
    void init(vk::Device device, vk::PhysicalDevice physicalDevice, MemoryAllocator* memAlloc,
              vk::DeviceSize stagingBlockSize = NVVK_DEFAULT_STAGING_BLOCKSIZE);

    void deinit();

    MemoryAllocator* getMemoryAllocator() {
        return m_memAlloc;
    }

    //--------------------------------------------------------------------------------------------------
    // Basic buffer creation
    Buffer createBuffer(const vk::BufferCreateInfo& info_,
                              const vk::MemoryPropertyFlags memUsage_ = vk::MemoryPropertyFlagBits::eDeviceLocal);

    //--------------------------------------------------------------------------------------------------
    // Simple buffer creation
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    Buffer createBuffer(vk::DeviceSize size_ = 0, vk::BufferUsageFlags usage_ = vk::BufferUsageFlags(),
                              const vk::MemoryPropertyFlags memUsage_ = vk::MemoryPropertyFlagBits::eDeviceLocal);

    //--------------------------------------------------------------------------------------------------
    // Simple buffer creation with data uploaded through staging manager
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    Buffer createBuffer(const vk::CommandBuffer& cmdBuf, const vk::DeviceSize& size_, const void* data_,
                              vk::BufferUsageFlags usage_,
                              vk::MemoryPropertyFlags memProps = vk::MemoryPropertyFlagBits::eDeviceLocal);

    //--------------------------------------------------------------------------------------------------
    // Simple buffer creation with data uploaded through staging manager
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    template <typename T>
    Buffer createBuffer(const vk::CommandBuffer& cmdBuf, const std::vector<T>& data_, vk::BufferUsageFlags usage_,
                              vk::MemoryPropertyFlags memProps_ = vk::MemoryPropertyFlagBits::eDeviceLocal) {
        return createBuffer(cmdBuf, sizeof(T) * data_.size(), data_.data(), usage_, memProps_);
    }

    //--------------------------------------------------------------------------------------------------
    // Basic image creation
    Image createImage(const vk::ImageCreateInfo& info_,
                            const vk::MemoryPropertyFlags memUsage_ = vk::MemoryPropertyFlagBits::eDeviceLocal);

    //--------------------------------------------------------------------------------------------------
    // Create an image with data uploaded through staging manager
    Image createImage(const vk::CommandBuffer& cmdBuf, size_t size_, const void* data_,
                            const vk::ImageCreateInfo& info_,
                            const vk::ImageLayout& layout_ = vk::ImageLayout::eShaderReadOnlyOptimal);

    //--------------------------------------------------------------------------------------------------
    // other variants could exist with a few defaults but we already have makeImage2DViewCreateInfo()
    // we could always override viewCreateInfo.image
    Texture createTexture(const Image& image, const vk::ImageViewCreateInfo& imageViewCreateInfo);
    Texture createTexture(const Image& image, const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                const vk::SamplerCreateInfo& samplerCreateInfo);

    //--------------------------------------------------------------------------------------------------
    // shortcut that creates the image for the texture
    // - creates the image
    // - creates the texture part by associating image and sampler
    //
    Texture createTexture(const vk::CommandBuffer& cmdBuf, size_t size_, const void* data_,
                                const vk::ImageCreateInfo& info_, const vk::SamplerCreateInfo& samplerCreateInfo,
                                const vk::ImageLayout& layout_ = vk::ImageLayout::eShaderReadOnlyOptimal,
                                bool isCube = false);

    AccelKHR createAcceleration(vk::AccelerationStructureCreateInfoKHR& accel_);

    //--------------------------------------------------------------------------------------------------
    // implicit staging operations triggered by create are managed here
    void finalizeStaging(vk::Fence fence = VK_NULL_HANDLE);
    void finalizeAndReleaseStaging(vk::Fence fence = VK_NULL_HANDLE);
    void releaseStaging();

    StagingMemoryManager* getStaging();
    const StagingMemoryManager* getStaging() const;

    //--------------------------------------------------------------------------------------------------
    // Destroy
    //
    void destroy(Buffer& b_);
    void destroy(Image& i_);
    void destroy(AccelKHR& a_);
    void destroy(Texture& t_);

    //--------------------------------------------------------------------------------------------------
    // Other
    //
    void* map(const Buffer& buffer);
    void unmap(const Buffer& buffer);
    void* map(const Image& image);
    void unmap(const Image& image);

    vk::Device getDevice() const {
        return m_device;
    }
    vk::PhysicalDevice getPhysicalDevice() const {
        return m_physicalDevice;
    }

  protected:
    // If necessary, these can be overriden to specialize the allocation, for instance to
    // enforce allocation of exportable
    virtual MemHandle AllocateMemory(const MemAllocateInfo& allocateInfo);
    virtual void CreateBufferEx(const vk::BufferCreateInfo& info_, vk::Buffer* buffer);
    virtual void CreateImageEx(const vk::ImageCreateInfo& info_, vk::Image* image);

    //--------------------------------------------------------------------------------------------------
    // Finding the memory type for memory allocation
    //
    uint32_t getMemoryType(uint32_t typeBits, const vk::MemoryPropertyFlags& properties);

    vk::Device m_device{VK_NULL_HANDLE};
    vk::PhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
    vk::PhysicalDeviceMemoryProperties m_memoryProperties{};
    MemoryAllocator* m_memAlloc{nullptr};
    std::unique_ptr<StagingMemoryManager> m_staging;
    SamplerPool& m_samplerPool;
};
} // namespace merian
