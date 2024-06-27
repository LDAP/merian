#pragma once

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "special_static_out.hpp"

namespace merian_nodes {

template <typename ValueType> class SpecialStaticIn;
template <typename ValueType>
using SpecialStaticInHandle = std::shared_ptr<SpecialStaticIn<ValueType>>;

// See corresponsing output.
template <typename ValueType>
class SpecialStaticIn
    : public TypedInputConnector<SpecialStaticOutHandle<ValueType>, const ValueType&> {
  public:
    SpecialStaticIn(const std::string& name)
        : TypedInputConnector<SpecialStaticOutHandle<ValueType>, const ValueType&>(name, 0) {}

    const ValueType& resource([[maybe_unused]] const GraphResourceHandle& resource) override {
        return debugable_ptr_cast<SpecialStaticOut<ValueType>>(resource)->connector_value;
    }

    void on_connect_output(const OutputConnectorHandle& output) override {
        auto casted_output = std::dynamic_pointer_cast<SpecialStaticOut<ValueType>>(output);
        if (!casted_output) {
            throw graph_errors::connector_error{fmt::format(
                "SpecialStaticIn {} cannot recive from {}.", Connector::name, output->name)};
        }
    }

    static SpecialStaticInHandle<ValueType> create(const std::string& name) {
        return std::make_shared<SpecialStaticIn<ValueType>>(name);
    }
};

} // namespace merian_nodes
