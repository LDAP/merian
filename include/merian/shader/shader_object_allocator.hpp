#pragma once

#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"

#include <memory>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

/**
 * @brief Allocator that manages descriptor set and buffer creation for shader objects.
 *
 * The allocator decides whether to create a new descriptor set or return a cached one
 * (e.g., cycling per frame-in-flight).
 */
class ShaderObjectAllocator {
  public:
    /**
     * @brief Allocate or retrieve a descriptor set for a shader object.
     *
     * @param object The shader object requesting a descriptor set
     * @param layout The descriptor set layout
     * @return A descriptor set (new or cached depending on implementation)
     */
    virtual DescriptorSetHandle allocate(const ShaderObjectHandle& object) = 0;

    /**
     * @brief Allocate a uniform buffer for ordinary shader data.
     */
    virtual BufferHandle allocate_uniform_buffer(vk::DeviceSize size) = 0;

    /**
     * @brief Get the staging memory manager for GPU uploads.
     */
    virtual StagingMemoryManagerHandle get_staging() const = 0;

    virtual ~ShaderObjectAllocator() = default;
};

using ShaderObjectAllocatorHandle = std::shared_ptr<ShaderObjectAllocator>;

/**
 * @brief Simple allocator that creates a new descriptor set every call.
 *
 * Suitable for one-shot usage or when the caller manages frame cycling externally.
 */
class SimpleShaderObjectAllocator : public ShaderObjectAllocator {
  public:
    SimpleShaderObjectAllocator(const ResourceAllocatorHandle& allocator);

    DescriptorSetHandle allocate(const ShaderObjectHandle& object) override;

    BufferHandle allocate_uniform_buffer(vk::DeviceSize size) override;

    StagingMemoryManagerHandle get_staging() const override;

  private:
    const ResourceAllocatorHandle allocator;
};

/**
 * @brief Allocator that caches descriptor sets per (object, frame-in-flight).
 *
 * Call set_iteration() before each frame to cycle to the correct descriptor set.
 * Returns cached sets for known objects at the current iteration index.
 */
class FrameCachingShaderObjectAllocator : public ShaderObjectAllocator {
  public:
    FrameCachingShaderObjectAllocator(const ResourceAllocatorHandle& allocator,
                                      uint32_t iterations_in_flight);

    DescriptorSetHandle allocate(const ShaderObjectHandle& object) override;

    BufferHandle allocate_uniform_buffer(vk::DeviceSize size) override;

    StagingMemoryManagerHandle get_staging() const override;

    void set_iteration(uint32_t iteration);

    void reset();

  private:
    const ResourceAllocatorHandle allocator;
    const uint32_t iterations_in_flight;
    uint32_t current_iteration = 0;

    std::unordered_map<ShaderObjectHandle, std::vector<DescriptorSetHandle>> sets;
};

} // namespace merian
