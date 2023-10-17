#pragma once

#include "merian/vk/memory/memory_allocator.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"
#include "merian/vk/sampler/sampler_pool.hpp"

#include <memory>
#include <optional>
#include <vulkan/vulkan.hpp>

namespace merian {

// A utility class to create and manage resources.
//
// Do not forget to finalize and release the resources from the staging memory manager that this
// class uses!
class ResourceAllocator : public std::enable_shared_from_this<ResourceAllocator> {
  public:
    ResourceAllocator(ResourceAllocator const&) = delete;
    ResourceAllocator& operator=(ResourceAllocator const&) = delete;
    ResourceAllocator() = delete;

    ResourceAllocator(const SharedContext& context,
                      const std::shared_ptr<MemoryAllocator>& memAllocator,
                      const std::shared_ptr<StagingMemoryManager> staging,
                      const std::shared_ptr<SamplerPool>& samplerPool);

    // All staging buffers must be cleared before
    virtual ~ResourceAllocator() {
        SPDLOG_DEBUG("destroy ResourceAllocator ({})", fmt::ptr(this));
    }

    //--------------------------------------------------------------------------------------------------

    std::shared_ptr<MemoryAllocator> getMemoryAllocator() {
        return m_memAlloc;
    }

    //--------------------------------------------------------------------------------------------------

    // Basic buffer creation
    BufferHandle createBuffer(const vk::BufferCreateInfo& info_,
                              const MemoryMappingType mapping_type = NONE,
                              const std::string& debug_name = {},
                              const std::optional<vk::DeviceSize> min_alignment = std::nullopt);

    // Simple buffer creation
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    BufferHandle createBuffer(const vk::DeviceSize size_,
                              const vk::BufferUsageFlags usage_ = {},
                              const MemoryMappingType mapping_type = NONE,
                              const std::string& debug_name = {},
                              const std::optional<vk::DeviceSize> min_alignment = std::nullopt);

    // Simple buffer creation with data uploaded through staging manager
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    BufferHandle createBuffer(const vk::CommandBuffer& cmdBuf,
                              const vk::DeviceSize& size_,
                              const vk::BufferUsageFlags usage_ = {},
                              const void* data_ = nullptr,
                              const MemoryMappingType mapping_type = NONE,
                              const std::string& debug_name = {},
                              const std::optional<vk::DeviceSize> min_alignment = std::nullopt);

    // Simple buffer creation with data uploaded through staging manager
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    template <typename T>
    BufferHandle createBuffer(const vk::CommandBuffer& cmdBuf,
                              const std::vector<T>& data_,
                              const vk::BufferUsageFlags usage_,
                              const std::string& debug_name = {},
                              const MemoryMappingType mapping_type = NONE,
                              const std::optional<vk::DeviceSize> min_alignment = std::nullopt) {
        return createBuffer(cmdBuf, sizeof(T) * data_.size(), usage_, data_.data(), mapping_type,
                            debug_name, min_alignment);
    }

    //--------------------------------------------------------------------------------------------------

    BufferHandle createScratchBuffer(const vk::DeviceSize size,
                                     const vk::DeviceSize alignment,
                                     const std::string& debug_name = {});

    //--------------------------------------------------------------------------------------------------

    // Basic image creation
    ImageHandle createImage(const vk::ImageCreateInfo& info_,
                            const MemoryMappingType mapping_type = NONE,
                            const std::string& debug_name = {});

    // Create an image with data uploaded through staging manager
    ImageHandle
    createImage(const vk::CommandBuffer& cmdBuf,
                const size_t size_,
                const void* data_,
                const vk::ImageCreateInfo& info_,
                const MemoryMappingType mapping_type = NONE,
                const vk::ImageLayout& layout_ = vk::ImageLayout::eShaderReadOnlyOptimal,
                const std::string& debug_name = {});

    //--------------------------------------------------------------------------------------------------

    TextureHandle createTexture(const ImageHandle& image,
                                const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                const vk::SamplerCreateInfo& samplerCreateInfo);

    // other variants could exist with a few defaults but we already have
    // makeImage2DViewCreateInfo() we could always override viewCreateInfo.image
    TextureHandle createTexture(const ImageHandle& image,
                                const vk::ImageViewCreateInfo& imageViewCreateInfo);

    // shortcut that creates the image for the texture
    // - creates the image
    // - creates the texture part by associating image and sampler
    TextureHandle
    createTexture(const vk::CommandBuffer& cmdBuf,
                  const size_t size_,
                  const void* data_,
                  const vk::ImageCreateInfo& info_,
                  const MemoryMappingType mapping_type,
                  const vk::SamplerCreateInfo& samplerCreateInfo,
                  const vk::ImageLayout& layout_ = vk::ImageLayout::eShaderReadOnlyOptimal,
                  const bool isCube = false,
                  const std::string& debug_name = {});

    //--------------------------------------------------------------------------------------------------

    AccelerationStructureHandle
    createAccelerationStructure(const vk::AccelerationStructureTypeKHR type,
                                const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                                const std::string& debug_name = {});

    //--------------------------------------------------------------------------------------------------

    std::shared_ptr<StagingMemoryManager> getStaging();
    const std::shared_ptr<StagingMemoryManager> getStaging() const;

    const SamplerPoolHandle& get_sampler_pool() const {
        return m_samplerPool;
    }

    //--------------------------------------------------------------------------------------------------

  protected:
    const SharedContext context;
    const std::shared_ptr<MemoryAllocator> m_memAlloc;
    const std::shared_ptr<StagingMemoryManager> m_staging;
    const SamplerPoolHandle m_samplerPool;
};

using ResourceAllocatorHandle = std::shared_ptr<ResourceAllocator>;

} // namespace merian
