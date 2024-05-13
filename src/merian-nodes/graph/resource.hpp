#pragma once

#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian_nodes {

using namespace merian;

class GraphResource {
  public:
    virtual ~GraphResource() = 0;

    // Allocate your resource. This is called exactly once per graph build. That means, depending
    // whether this resource is reused between builds it might be called multiple times.
    //
    // If the resource is availible via descriptor you must enure needs_descriptor_update() == true
    // after this call.
    //
    // If using aliasing_allocator the graph might alias the underlying memory, meaning it is only
    // valid when the node is executed and no guarantees about the contents of the memory can be
    // made. However, it is guaranteed between calls to connector.on_pre_process and
    // connector.on_post_process with this resource the memory is not in use and syncronization is
    // ensured.
    virtual void allocate(const ResourceAllocator& allocator,
                          const ResourceAllocator& aliasing_allocator);

    // Signalize that the resource has changed and descriptor set updates are necessary.
    // You can assume that after this method returns true the descriptor sets are updated (and you
    // can reset needs_descriptor_update).
    virtual bool needs_descriptor_update() const {
        throw std::runtime_error{"resource is not accessible using a descriptor"};
    }

    // Write the descriptor update to the specified binding (please).
    // This might be called multiple times after needs_descriptor_update() is set to true (for every
    // resource). This is only called if get_descriptor_info() != std::nullopt but
    // needs_descriptor_update() can also return false.
    virtual void get_descriptor_update([[maybe_unused]] const GraphResource& resource,
                                       [[maybe_unused]] const uint32_t binding,
                                       [[maybe_unused]] DescriptorSetUpdate& update) {
        throw std::runtime_error{"resource is not accessible using a descriptor"};
    }
};

using GraphResourceHandle = std::shared_ptr<GraphResource>;

} // namespace merian_nodes
