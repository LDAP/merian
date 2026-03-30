#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"

#include <memory>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

/**
 * @brief Allocator that manages descriptor container and buffer creation for shader objects.
 *
 * The allocator decides whether to create a new descriptor container or return a cached one
 * (e.g., cycling per frame-in-flight).
 */
class ShaderObjectAllocator {
  public:
    virtual DescriptorContainerHandle allocate(ShaderObject* object) = 0;

    virtual void free(ShaderObject* object) = 0;

    virtual BufferHandle allocate_uniform_buffer(vk::DeviceSize size) = 0;

    virtual StagingMemoryManagerHandle get_staging() const = 0;

    virtual ~ShaderObjectAllocator() = default;
};

using ShaderObjectAllocatorHandle = std::shared_ptr<ShaderObjectAllocator>;

/**
 * @brief Simple allocator that creates a new descriptor set every call.
 */
class SimpleShaderObjectAllocator : public ShaderObjectAllocator {
  public:
    SimpleShaderObjectAllocator(const ResourceAllocatorHandle& allocator);

    DescriptorContainerHandle allocate(ShaderObject* object) override;

    void free(ShaderObject* object) override;

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

    DescriptorContainerHandle allocate(ShaderObject* object) override;

    void free(ShaderObject* object) override;

    BufferHandle allocate_uniform_buffer(vk::DeviceSize size) override;

    StagingMemoryManagerHandle get_staging() const override;

    void set_iteration(uint32_t iteration);

    void reset();

  private:
    const ResourceAllocatorHandle allocator;
    const uint32_t iterations_in_flight;
    uint32_t current_iteration = 0;

    std::unordered_map<ShaderObject*, std::vector<DescriptorSetHandle>> cache;
};

} // namespace merian
