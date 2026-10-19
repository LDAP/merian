#pragma once

#include "connector.hpp"

#include "merian/vk/command/submission.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

class InputConnector;
using InputConnectorHandle = std::shared_ptr<InputConnector>;

struct ResourceCreateInfo {
    // The inputs it feeds, also through versions written in place.
    const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs;
    // The union of the declared accesses of this connector and all connected inputs: allocate with
    // its usage flags.
    ConnectorAccess combined_access;
    // Memory for this copy, not bound before on_connected.
    ResourceAllocatorHandle allocator;
    // The node relies on the contents surviving iterations.
    bool persistent;
    // Copy c is written in the iterations i with i % copy_count == c.
    uint32_t copy_index;
    uint32_t copy_count;
    std::string debug_name;
};

struct ResourceConnectInfo {
    // Of the producing node, for views, descriptors and uploads.
    ResourceAllocatorHandle allocator;
    uint32_t copy_index;
};

// The base class for all output connectors.
class OutputConnector : public Connector {
  public:
    // Called once per copy and connect.
    virtual GraphResourceHandle create_resource(const ResourceCreateInfo& info) = 0;

    // Called once per copy after the memory of all copies is bound: create views and query device
    // addresses here. Recorded commands are submitted and awaited before the first run; what they
    // write is kept for a delayed input that reads a copy before its first write.
    virtual void on_connected([[maybe_unused]] const ResourceConnectInfo& info,
                              [[maybe_unused]] Submission& submission,
                              [[maybe_unused]] const GraphResourceHandle& resource) {}

    // A delayed input can read it: create_resource makes independent resources.
    virtual bool supports_delay() const {
        return true;
    }

    // An in-place input can write the resource and pass it on.
    virtual bool supports_in_place() const {
        return false;
    }

    virtual void properties(Properties& config) {
        config.output_text(fmt::format("supports delay: {}", supports_delay()));
    }
};

using OutputConnectorHandle = std::shared_ptr<OutputConnector>;

} // namespace merian
