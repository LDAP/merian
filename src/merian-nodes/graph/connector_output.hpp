#pragma once

#include "connector.hpp"

#include "merian/vk/memory/resource_allocator.hpp"

namespace merian_nodes {

class InputConnector;
using InputConnectorHandle = std::shared_ptr<InputConnector>;

// Do not inherit from this class, inherit from TypedOutputConnector instead.
class OutputConnector : public Connector {
  public:
    OutputConnector(const std::string& name, const bool supports_delay)
        : Connector(name), supports_delay(supports_delay) {}

    // Create the resource for this output. This is called max_delay + 1 times per graph build.
    // Throw connector_error if this output cannot output to one of the supplied inputs.
    //
    // If the resource is availible via descriptors you must ensure needs_descriptor_update.
    //
    // If using aliasing_allocator the graph might alias the underlying memory, meaning it is only
    // valid when the node is executed and no guarantees about the contents of the memory can be
    // made. However, it is guaranteed between calls to connector.on_pre_process and
    // connector.on_post_process with this resource the memory is not in use and syncronization is
    // ensured.
    virtual GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator) = 0;

    virtual void configuration(Configuration& config) {
        config.output_text(fmt::format("supports delay: {}", supports_delay));
    }

  public:
    const bool supports_delay;
};

using OutputConnectorHandle = std::shared_ptr<OutputConnector>;

/**
 * @brief      The base class for all output connectors.
 *
 * @tparam     ResourceAccessType  defines how nodes can access the underlying resource of this
 * connector. If the type is void, access is not possible.
 */
template <typename ResourceAccessType = void> class TypedOutputConnector : public OutputConnector {
  public:
    TypedOutputConnector(const std::string& name, const bool supports_delay)
        : OutputConnector(name, supports_delay) {}

    using resource_access_type = ResourceAccessType;

    virtual ResourceAccessType resource(const GraphResourceHandle& resource) = 0;
};

template <typename ResourceType, typename ResourceAccessType = void>
using TypedOutputConnectorHandle = std::shared_ptr<TypedOutputConnector<ResourceAccessType>>;

} // namespace merian_nodes
