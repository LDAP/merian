#pragma once

#include "connector_output.hpp"

namespace merian_nodes {

// Do not inherit from this class, inherit from TypedInputConnector instead.
class InputConnector : public Connector {

  public:
    InputConnector(const std::string& name, const uint32_t delay) : Connector(name), delay(delay) {}

    virtual void configuration(Configuration& config) {
        config.output_text(fmt::format("delay: {}", delay));
    }
  public:
    // The number of iterations the corresponding resource is accessed later.
    const uint32_t delay;
};

using InputConnectorHandle = std::shared_ptr<InputConnector>;

/**
 * @brief      The base class for all input connectors.
 *
 * @tparam     The corresponding output connector type.
 * @tparam     ResourceAccessType  defines how nodes can access the underlying resource of this
 * connector. If the type is void, access is not possible.
 */
template <typename OutputConnectorType, typename ResourceAccessType = void>
class TypedInputConnector : public InputConnector {
  public:
    using resource_access_type = ResourceAccessType;
    using output_connector_type = OutputConnectorType;

    TypedInputConnector(const std::string& name, const uint32_t delay)
        : InputConnector(name, delay) {}

    virtual ResourceAccessType resource(const GraphResourceHandle& resource) = 0;
};

template <typename OutputConnectorType, typename ResourceAccessType = void>
using TypedInputConnectorHandle =
    std::shared_ptr<TypedInputConnector<OutputConnectorType, ResourceAccessType>>;

} // namespace merian_nodes
