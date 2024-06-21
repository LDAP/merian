#pragma once

#include "merian/vk/extension/extension_vk_debug_utils.hpp"
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
//
// Debug names are forwarded to the memory allocator. If NDEBUG is not defined the debug are
// attempted to be set using the debug extension.
class ResourceAllocator : public std::enable_shared_from_this<ResourceAllocator> {
  public:
    ResourceAllocator(ResourceAllocator const&) = delete;
    ResourceAllocator& operator=(ResourceAllocator const&) = delete;
    ResourceAllocator() = delete;

    ResourceAllocator(const SharedContext& context,
                      const std::shared_ptr<MemoryAllocator>& memAllocator,
                      const StagingMemoryManagerHandle staging,
                      const SamplerPoolHandle& samplerPool);

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
                              const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                              const std::string& debug_name = {},
                              const std::optional<vk::DeviceSize> min_alignment = std::nullopt);

    // Simple buffer creation
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    BufferHandle createBuffer(const vk::DeviceSize size_,
                              const vk::BufferUsageFlags usage_ = {},
                              const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                              const std::string& debug_name = {},
                              const std::optional<vk::DeviceSize> min_alignment = std::nullopt);

    // Simple buffer creation with data uploaded through staging manager
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    BufferHandle createBuffer(const vk::CommandBuffer& cmdBuf,
                              const vk::DeviceSize& size_,
                              const vk::BufferUsageFlags usage_ = {},
                              const void* data_ = nullptr,
                              const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                              const std::string& debug_name = {},
                              const std::optional<vk::DeviceSize> min_alignment = std::nullopt);

    // Simple buffer creation with data uploaded through staging manager
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    template <typename T>
    BufferHandle createBuffer(const vk::CommandBuffer& cmdBuf,
                              const std::vector<T>& data_,
                              const vk::BufferUsageFlags usage_,
                              const std::string& debug_name = {},
                              const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                              const std::optional<vk::DeviceSize> min_alignment = std::nullopt) {
        return createBuffer(cmdBuf, sizeof(T) * data_.size(), usage_, data_.data(), mapping_type,
                            debug_name, min_alignment);
    }

    // Utility function that creates a larger buffer if optional_existing_buffer is to small or
    // null.
    //
    // Use a gowth factor >= 1 to ensure exponential growth.
    //
    // returns true if the buffer was (re)created and the buffer handle was updated, false if the
    // existing buffer can be used.
    bool ensureBufferSize(BufferHandle& buffer,
                          const vk::DeviceSize buffer_size,
                          const vk::BufferUsageFlags usage,
                          const std::string& debug_name = {},
                          const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                          const std::optional<vk::DeviceSize> min_alignment = std::nullopt,
                          const float growth_factor = 1) {
        assert(growth_factor >= 1);

        if (buffer && buffer->get_size() >= buffer_size) {
            return false;
        }

        buffer = createBuffer(buffer_size * growth_factor, usage, mapping_type, debug_name,
                              min_alignment);
        return true;
    }

    // Returns a dummy buffer containing exactly 4 entries of the "missing texture" color
    // (1,0,1,1).
    const BufferHandle& get_dummy_buffer() const;

    //--------------------------------------------------------------------------------------------------

    // Create a scratch buffer for acceleration buffer builds.
    BufferHandle createScratchBuffer(const vk::DeviceSize size,
                                     const vk::DeviceSize alignment,
                                     const std::string& debug_name = {});

    // Create a buffer that holds acceleration strcture instances.
    BufferHandle createInstancesBuffer(const uint32_t instance_count,
                                       const std::string& debug_name = {});

    //--------------------------------------------------------------------------------------------------

    // Basic image creation
    ImageHandle createImage(const vk::ImageCreateInfo& info_,
                            const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                            const std::string& debug_name = {});

    // Create an image with data uploaded through staging manager
    //
    // Important: You are responsible to insert a barrier for the upload.
    ImageHandle createImage(const vk::CommandBuffer& cmdBuf,
                            const size_t size_,
                            const void* data_,
                            const vk::ImageCreateInfo& info_,
                            const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                            const std::string& debug_name = {});

    //--------------------------------------------------------------------------------------------------

    TextureHandle createTexture(const ImageHandle& image,
                                const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                const SamplerHandle& sampler,
                                const std::string& debug_name = {});

    TextureHandle createTexture(const ImageHandle& image,
                                const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                const vk::SamplerCreateInfo& samplerCreateInfo,
                                const std::string& debug_name = {});

    // Create a texture with a linear sampler if the view format supports it.
    // With a view to the whole subresource (using image->make_view_create_info()).
    TextureHandle createTexture(const ImageHandle& image, const std::string& debug_name = {});

    // Create a texture with a linear sampler if the view format supports it.
    TextureHandle createTexture(const ImageHandle& image,
                                const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                const std::string& debug_name = {});

    // shortcut to create a texture from RGB8 data.
    // layout: the layout for the image view
    //
    // Important: You are responsible to perform the image transition!
    TextureHandle createTextureFromRGBA8(const vk::CommandBuffer& cmd,
                                         const uint32_t* data,
                                         const uint32_t width,
                                         const uint32_t height,
                                         const vk::Filter filter,
                                         const bool isSRGB = true,
                                         const std::string& debug_name = {});

    // Returns a dummy 4x4 texture with the "missing texture" color (1,0,1,1).
    const TextureHandle& get_dummy_texture() const;

    //--------------------------------------------------------------------------------------------------

    AccelerationStructureHandle
    createAccelerationStructure(const vk::AccelerationStructureTypeKHR type,
                                const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                                const std::string& debug_name = {});

    //--------------------------------------------------------------------------------------------------

    StagingMemoryManagerHandle getStaging();

    const StagingMemoryManagerHandle& getStaging() const;

    const SamplerPoolHandle& get_sampler_pool() const {
        return m_samplerPool;
    }

    //--------------------------------------------------------------------------------------------------

  protected:
    const SharedContext context;
    const std::shared_ptr<MemoryAllocator> m_memAlloc;
    const StagingMemoryManagerHandle m_staging;
    const SamplerPoolHandle m_samplerPool;
    const std::shared_ptr<ExtensionVkDebugUtils> debug_utils;

    TextureHandle dummy_texture;
    BufferHandle dummy_buffer;
};

using ResourceAllocatorHandle = std::shared_ptr<ResourceAllocator>;

} // namespace merian
