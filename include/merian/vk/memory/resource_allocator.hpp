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

    ResourceAllocator(const ContextHandle& context,
                      const MemoryAllocatorHandle& memAllocator,
                      const StagingMemoryManagerHandle& staging,
                      const SamplerPoolHandle& samplerPool,
                      const DescriptorSetAllocatorHandle& descriptor_pool);

    // All staging buffers must be cleared before
    virtual ~ResourceAllocator() {
        SPDLOG_DEBUG("destroy ResourceAllocator ({})", fmt::ptr(this));
    }

    //--------------------------------------------------------------------------------------------------

    std::shared_ptr<MemoryAllocator> get_memory_allocator() {
        return m_memAlloc;
    }

    //--------------------------------------------------------------------------------------------------

    // Basic buffer creation
    BufferHandle create_buffer(const vk::BufferCreateInfo& info_,
                               const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                               const std::string& debug_name = {},
                               const std::optional<vk::DeviceSize> min_alignment = std::nullopt);

    // Simple buffer creation
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    BufferHandle create_buffer(const vk::DeviceSize size_,
                               const vk::BufferUsageFlags usage_ = {},
                               const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                               const std::string& debug_name = {},
                               const std::optional<vk::DeviceSize> min_alignment = std::nullopt);

    // Simple buffer creation with data uploaded through staging manager
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    BufferHandle create_buffer(const CommandBufferHandle& cmdBuf,
                               const vk::DeviceSize& size_,
                               const vk::BufferUsageFlags usage_ = {},
                               const void* data_ = nullptr,
                               const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                               const std::string& debug_name = {},
                               const std::optional<vk::DeviceSize> min_alignment = std::nullopt);

    // Simple buffer creation with data uploaded through staging manager
    // implicitly sets VK_BUFFER_USAGE_TRANSFER_DST_BIT
    template <typename T>
    BufferHandle create_buffer(const CommandBufferHandle& cmdBuf,
                               const std::vector<T>& data_,
                               const vk::BufferUsageFlags usage_,
                               const std::string& debug_name = {},
                               const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                               const std::optional<vk::DeviceSize> min_alignment = std::nullopt) {
        return create_buffer(cmdBuf, sizeof(T) * data_.size(), usage_, data_.data(), mapping_type,
                             debug_name, min_alignment);
    }

    // Utility function that creates a larger buffer if optional_existing_buffer is to small or
    // null.
    //
    // Use a gowth factor >= 1 to ensure exponential growth.
    //
    // returns true if the buffer was (re)created and the buffer handle was updated, false if the
    // existing buffer can be used.
    bool ensure_buffer_size(BufferHandle& buffer,
                            const vk::DeviceSize buffer_size,
                            const vk::BufferUsageFlags usage,
                            const std::string& debug_name = {},
                            const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                            const std::optional<vk::DeviceSize> min_alignment = std::nullopt,
                            const float growth_factor = 1);

    // Returns a dummy buffer containing exactly 4 entries of the "missing texture" color
    // (1,0,1,1).
    const BufferHandle& get_dummy_buffer() const;

    //--------------------------------------------------------------------------------------------------

    // Create a scratch buffer for acceleration buffer builds.
    BufferHandle create_scratch_buffer(const vk::DeviceSize size,
                                       const vk::DeviceSize alignment,
                                       const std::string& debug_name = {});

    // Create a buffer that holds acceleration strcture instances.
    BufferHandle create_instances_buffer(const uint32_t instance_count,
                                         const std::string& debug_name = {});

    //--------------------------------------------------------------------------------------------------

    // Basic image creation
    ImageHandle create_image(const vk::ImageCreateInfo& info_,
                             const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                             const std::string& debug_name = {});

    // Create an image with data uploaded through staging manager
    //
    // Important: You are responsible to insert a barrier for the upload.
    ImageHandle create_image(const CommandBufferHandle& cmdBuf,
                             const void* data_,
                             const vk::ImageCreateInfo& info_,
                             const MemoryMappingType mapping_type = MemoryMappingType::NONE,
                             const std::string& debug_name = {});

    ImageHandle create_image_from_rgba8(const CommandBufferHandle& cmd,
                                        const uint32_t* data,
                                        const uint32_t width,
                                        const uint32_t height,
                                        const vk::ImageUsageFlags usage,
                                        const bool isSRGB = true,
                                        const uint32_t mip_levels = 1,
                                        const std::string& debug_name = {});

    // Returns a dummy 4x4 image with the "missing texture" color (1,0,1,1).
    const ImageViewHandle& get_dummy_storage_image_view() const;

    //--------------------------------------------------------------------------------------------------

    ImageViewHandle create_image_view(const ImageHandle& image,
                                      const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                      const std::string& debug_name = {});

    //--------------------------------------------------------------------------------------------------

    // shortcut to create an image view and a texture
    TextureHandle create_texture(const ImageHandle& image,
                                 const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                 const SamplerHandle& sampler,
                                 const std::string& debug_name = {});

    // shortcut to create an image view and a texture
    TextureHandle create_texture(const ImageHandle& image,
                                 const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                 const vk::SamplerCreateInfo& samplerCreateInfo,
                                 const std::string& debug_name = {});

    // shortcut to create an image view and a texture
    // Create a texture with a linear sampler if the view format supports it.
    // With a view to the whole subresource (using image->make_view_create_info()).
    TextureHandle create_texture(const ImageHandle& image, const std::string& debug_name = {});

    // shortcut to create an image view and a texture
    // Create a texture with a linear sampler if the view format supports it.
    TextureHandle create_texture(const ImageHandle& image,
                                 const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                 const std::string& debug_name = {});

    // shortcut to create an image view and a texture from RGB8 data
    // layout: the layout for the image view
    //
    // Important: You are responsible to perform the image transition!
    TextureHandle create_texture_from_rgba8(const CommandBufferHandle& cmd,
                                            const uint32_t* data,
                                            const uint32_t width,
                                            const uint32_t height,
                                            const vk::Filter mag_filter,
                                            const vk::Filter min_filter,
                                            const bool isSRGB = true,
                                            const std::string& debug_name = {},
                                            const bool generate_mipmaps = false,
                                            const vk::ImageUsageFlags additional_usage_flags = {});

    // Returns a dummy 4x4 texture with the "missing texture" color (1,0,1,1).
    const TextureHandle& get_dummy_texture() const;

    //--------------------------------------------------------------------------------------------------

    AccelerationStructureHandle
    create_acceleration_structure(const vk::AccelerationStructureTypeKHR type,
                                  const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                                  const std::string& debug_name = {});

    //--------------------------------------------------------------------------------------------------

    // Shortcut for get_descriptor_pool()->allocate(...)
    DescriptorSetHandle allocate_descriptor_set(const DescriptorSetLayoutHandle& layout);

    // Shortcut for get_descriptor_pool()->allocate(...)
    std::vector<DescriptorSetHandle>
    allocate_descriptor_set(const DescriptorSetLayoutHandle& layout, const uint32_t set_count);

    //--------------------------------------------------------------------------------------------------

    StagingMemoryManagerHandle get_staging();

    const StagingMemoryManagerHandle& get_staging() const;

    const SamplerPoolHandle& get_sampler_pool() const {
        return m_samplerPool;
    }

    const DescriptorSetAllocatorHandle& get_descriptor_pool() {
        return descriptor_pool;
    }

    const ContextHandle& get_context() const {
        return context;
    }

    //--------------------------------------------------------------------------------------------------

  protected:
    const ContextHandle context;
    const std::shared_ptr<MemoryAllocator> m_memAlloc;
    const StagingMemoryManagerHandle m_staging;
    const SamplerPoolHandle m_samplerPool;
    const DescriptorSetAllocatorHandle descriptor_pool;
    const std::shared_ptr<ExtensionVkDebugUtils> debug_utils;

    ImageViewHandle dummy_storage_image_view;
    TextureHandle dummy_texture;
    BufferHandle dummy_buffer;
};

using ResourceAllocatorHandle = std::shared_ptr<ResourceAllocator>;

} // namespace merian
