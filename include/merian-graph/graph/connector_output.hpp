#pragma once

#include "connector.hpp"

#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

class InputConnector;
using InputConnectorHandle = std::shared_ptr<InputConnector>;

// The base class for all output connectors.
class OutputConnector : public Connector {
  public:
    OutputConnector(const bool supports_delay) : supports_delay(supports_delay) {}

    // Create the resource for this output. This is called max_delay + 1 times per graph build.
    //
    // If the resource is availible via descriptors you must ensure needs_descriptor_update.
    //
    // If using aliasing_allocator the graph might alias the underlying memory, meaning it is only
    // valid when the node is executed and no guarantees about the contents of the memory can be
    // made. However, it is guaranteed between calls to connector.on_pre_process and
    // connector.on_post_process with this resource the memory is not in use and syncronization is
    // ensured.
    //
    // The inputs are supplied in the order they are serialized by the graph. combined_access is
    // the union of the declared ConnectorAccess of this port and all connected inputs (use its
    // usage flags for allocation).
    //
    // resource_index: 0 <= i <= max_delay
    // ring_size: Number of iterations in flight
    virtual GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ConnectorAccess& combined_access,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resource_index,
                    const uint32_t ring_size) = 0;

    // Throw invalid_connection, if the resource cannot interface with the supplied connector (try
    // dynamic cast or use merian::test_shared_ptr_types). Can also be used to pre-compute barriers
    // or similar.
    virtual void on_connect_input([[maybe_unused]] const InputConnectorHandle& input) {}

    // Called once after the graph is connected with all resources of this output (one per ring
    // slot). Record one-time device initialization on cmd; it is submitted and awaited before the
    // first run.
    virtual void on_connected([[maybe_unused]] const CommandBufferHandle& cmd,
                              [[maybe_unused]] const std::vector<GraphResourceHandle>& resources) {}

    virtual void properties(Properties& config) {
        config.output_text(fmt::format("supports delay: {}", supports_delay));
    }

  public:
    const bool supports_delay;
};

using OutputConnectorHandle = std::shared_ptr<OutputConnector>;

} // namespace merian
