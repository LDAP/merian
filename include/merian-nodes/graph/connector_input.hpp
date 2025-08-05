#pragma once

#include "connector_output.hpp"

#include "merian/utils/pointer.hpp"

namespace merian_nodes {

// The base class for all input connectors.
class InputConnector : public Connector {

  public:
    InputConnector(const std::string& name, const uint32_t delay, const bool optional = false)
        : Connector(name), delay(delay), optional(optional) {}

    virtual void properties(Properties& config) {
        config.output_text(fmt::format("delay: {}\noptional: {}", delay, optional));
    }

    // Throw invalid_connection, if the resource cannot interface with the supplied connector (try
    // dynamic cast or use merian::test_shared_ptr_types). Can also be used to pre-compute barriers
    // or similar.
    virtual void on_connect_output([[maybe_unused]] const OutputConnectorHandle& output) {}

  public:
    // The number of iterations the corresponding resource is accessed later.
    const uint32_t delay;
    const bool optional;
};

using InputConnectorHandle = std::shared_ptr<InputConnector>;

/**
 * @brief      Mixin for input connectors that allows accessing the connected output.
 *
 * For optional inputs only the descriptor related methods are called to provide a dummy
 * binding.
 *
 * @tparam     The corresponding output connector type.
 * @tparam     ResourceAccessType  defines how nodes can access the underlying resource of this
 * connector. If the type is void, access is not possible.
 */
template <typename OutputConnectorType> class OutputAccessibleInputConnector {
  public:
    using output_connector_type = OutputConnectorType;

    OutputAccessibleInputConnector() {}

    virtual OutputConnectorType output_connector(const OutputConnectorHandle& output) const {
        return debugable_ptr_cast<typename OutputConnectorType::element_type>(output);
    }
};

template <typename OutputConnectorType, typename ResourceAccessType = void>
using OutputAccessibleInputConnectorHandle =
    std::shared_ptr<OutputAccessibleInputConnector<OutputConnectorType>>;

} // namespace merian_nodes
