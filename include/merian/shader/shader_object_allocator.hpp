#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <memory>
#include <unordered_map>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

struct ShaderObjectAllocation {
    DescriptorContainerHandle container;
    // The container has not seen the object's writes yet and must be initialized by replay.
    bool freshly_allocated;
};

class ShaderObjectAllocator {
  public:
    // shared_ptr so cache-backed implementations can hold a weak_ptr for liveness.
    virtual ShaderObjectAllocation allocate(const ShaderObjectHandle& object) = 0;

    virtual ~ShaderObjectAllocator() = default;
};

using ShaderObjectAllocatorHandle = std::shared_ptr<ShaderObjectAllocator>;

class SimpleShaderObjectAllocator : public ShaderObjectAllocator {
  public:
    SimpleShaderObjectAllocator(const ResourceAllocatorHandle& allocator);

    ShaderObjectAllocation allocate(const ShaderObjectHandle& object) override;

  private:
    const ResourceAllocatorHandle allocator;
};

class FrameCachingShaderObjectAllocator : public ShaderObjectAllocator {
  public:
    FrameCachingShaderObjectAllocator(const ResourceAllocatorHandle& allocator,
                                      uint32_t iterations_in_flight);

    ShaderObjectAllocation allocate(const ShaderObjectHandle& object) override;

    // Call before each frame to cycle.
    void set_iteration(uint32_t iteration);

    void reset();

  private:
    void prune_expired();

    const ResourceAllocatorHandle allocator;
    const uint32_t iterations_in_flight;
    uint32_t current_iteration = 0;

    struct Entry {
        std::weak_ptr<ShaderObject> live;
        std::vector<DescriptorSetHandle> sets;
        // Each per-iteration set is allocated empty and must be replayed on its first use.
        std::vector<bool> replayed;
    };
    std::unordered_map<ShaderObject*, Entry> cache;
};

} // namespace merian
