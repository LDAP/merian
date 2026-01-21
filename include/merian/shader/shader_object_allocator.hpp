#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <memory>

namespace merian {

class ShaderObject;
using ShaderObjectHandle = std::shared_ptr<ShaderObject>;

/**
 * @brief Allocator that manages descriptor set creation and caching for shader objects.
 *
 */
class ShaderObjectAllocator {
  public:
    /**
     * @brief Get or create a descriptor set for a shader object.
     *
     * @param object The shader object needing a descriptor set
     * @param layout The descriptor set layout
     * @return Cached or newly created descriptor set
     */
    virtual DescriptorContainerHandle
    get_or_create_descriptor_set(const ShaderObjectHandle& object) = 0;
};

using ShaderObjectAllocatorHandle = std::shared_ptr<ShaderObjectAllocator>;

class DescriptorSetShaderObjectAllocator : public ShaderObjectAllocator {
  public:
    DescriptorSetShaderObjectAllocator(const ResourceAllocatorHandle& allocator,
                                       const uint32_t iterations_in_flight);

    DescriptorContainerHandle
    get_or_create_descriptor_set(const ShaderObjectHandle& object) override;

    void set_iteration(const uint32_t iteration);

    void reset();

  private:
    const ResourceAllocatorHandle allocator;
    const uint32_t iterations_in_flight;
    uint32_t iteration_in_flight;

    std::unordered_map<ShaderObjectHandle, std::vector<DescriptorSetHandle>> sets;
};

} // namespace merian
