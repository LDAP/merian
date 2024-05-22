#pragma once

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/graph/connector_output.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian_nodes {

using namespace merian;

class GraphResource {
  public:
    virtual ~GraphResource() = 0;

    // Throw connector_error, if the resource cannot interface with the supplied connector (try dynamic cast
    // or use merian::test_shared_ptr_types). Can also be used to pre-compute barriers or similar.
    // This is called for every input that accesses the resource.
    virtual void on_connect_input([[maybe_unused]] const InputConnectorHandle& input) {}

    // Throw connector_error,, if the resource cannot interface with the supplied connector (try dynamic cast
    // or use merian::test_shared_ptr_types). Can also be used to pre-compute barriers or similar.
    // This is called exactly once for the output that created the resource.
    virtual void on_connect_output([[maybe_unused]] const OutputConnectorHandle& input) {}

    // Allocate your resource. This is called exactly once per graph build. That means, depending
    // whether this resource is reused between builds it might be called multiple times.
    // It is guaranteed that allocate() is called after all on_connect_* callbacks.
    //
    // If the resource is availible via descriptors you must ensure needs_descriptor_update() ==
    // true after this call.
    //
    // If using aliasing_allocator the graph might alias the underlying memory, meaning it is only
    // valid when the node is executed and no guarantees about the contents of the memory can be
    // made. However, it is guaranteed between calls to connector.on_pre_process and
    // connector.on_post_process with this resource the memory is not in use and syncronization is
    // ensured.
    virtual void allocate([[maybe_unused]] const ResourceAllocatorHandle& allocator,
                          [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator) {}
};

using GraphResourceHandle = std::shared_ptr<GraphResource>;

} // namespace merian_nodes
