#pragma once

#include "connector_output.hpp"

namespace merian_nodes {

// Do not inherit from this class, inherit from TypedInputConnector instead.
class InputConnector : public Connector {

  public:
    InputConnector(const std::string& name, const uint32_t delay) : Connector(name), delay(delay) {}

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
    virtual ResourceAccessType resource(GraphResourceHandle& resource) = 0;
};

template <typename OutputConnectorType, typename ResourceAccessType = void>
using TypedInputConnectorHandle =
    std::shared_ptr<TypedInputConnector<OutputConnectorType, ResourceAccessType>>;

// Access the outputs that are connected to your inputs.
class ConnectorIOMap {
  public:
    ConnectorIOMap(const std::function<OutputConnectorHandle(const InputConnectorHandle&)>&
                       output_for_input)
        : output_for_input(output_for_input) {}

    template <typename OutputConnectorType>
    const std::shared_ptr<OutputConnectorType>&
    operator[](const TypedInputConnectorHandle<OutputConnectorType> input_connector) {
        return std::static_pointer_cast<OutputConnectorType>(output_for_input(input_connector));
    }

  private:
    const std::function<OutputConnectorHandle(const InputConnectorHandle&)>&
        output_for_input;
};

class ConnectorResourceMap {};

} // namespace merian_nodes
