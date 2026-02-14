#pragma once

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "special_static_out.hpp"

namespace merian {

template <typename ValueType> class SpecialStaticIn;
template <typename ValueType>
using SpecialStaticInHandle = std::shared_ptr<SpecialStaticIn<ValueType>>;

// See corresponsing output.
template <typename ValueType>
class SpecialStaticIn : public InputConnector,
                        public OutputAccessibleInputConnector<SpecialStaticOutHandle<ValueType>>,
                        public AccessibleConnector<const ValueType&> {
  public:
    SpecialStaticIn(const bool optional = false) : InputConnector(0, optional) {}

    const ValueType& resource([[maybe_unused]] const GraphResourceHandle& resource) override {
        return debugable_ptr_cast<SpecialStaticOut<ValueType>>(resource)->value();
    }

    void on_connect_output(const OutputConnectorHandle& output) override {
        auto casted_output = std::dynamic_pointer_cast<SpecialStaticOut<ValueType>>(output);
        if (!casted_output) {
            throw graph_errors::invalid_connection{"SpecialStaticIn cannot receive from output."};
        }
    }

    static SpecialStaticInHandle<ValueType> create(const bool optional = false) {
        return std::make_shared<SpecialStaticIn<ValueType>>(optional);
    }
};

} // namespace merian
