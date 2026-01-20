#pragma once

#include "merian/vk/descriptors/descriptor_container.hpp"
#include "merian/vk/descriptors/descriptor_set_layout.hpp"

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
    get_or_create_descriptor_set(const ShaderObjectHandle& object,
                                 const DescriptorSetLayoutHandle& layout) = 0;
};

using ShaderObjectAllocatorHandle = std::shared_ptr<ShaderObjectAllocator>;

} // namespace merian
