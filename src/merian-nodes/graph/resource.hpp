#pragma once

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/graph/connector_output.hpp"

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
};

using GraphResourceHandle = std::shared_ptr<GraphResource>;

} // namespace merian_nodes
