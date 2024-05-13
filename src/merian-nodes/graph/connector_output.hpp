#pragma once

#include "connector.hpp"
#include "resource.hpp"

namespace merian_nodes {

// Do not inherit from this class, inherit from TypedOutputConnector instead.
class OutputConnector : public Connector {
  public:
    OutputConnector(const std::string& name, const bool supports_delay)
        : Connector(name), supports_delay(supports_delay) {}

    // Create the resource for this output.
    // This is called exactly once for every graph build, if supports_delay == false. Otherwise it
    // might be called multiple times.
    virtual GraphResourceHandle create_resource() = 0;

  public:
    const bool supports_delay;
};

using OutputConnectorHandle = std::vector<std::shared_ptr<OutputConnector>>;

/**
 * @brief      The base class for all output connectors.
 *
 * @tparam     ResourceAccessType  defines how nodes can access the underlying resource of this
 * connector. If the type is void, access is not possible.
 */
template <typename ResourceAccessType = void> class TypedOutputConnector : public OutputConnector {
  public:
    virtual ResourceAccessType resource(GraphResourceHandle& resource) = 0;
};

template <typename ResourceAccessType = void>
using TypedOutputConnectorHandle =
    std::vector<std::shared_ptr<TypedOutputConnector<ResourceAccessType>>>;

} // namespace merian_nodes
