#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "special_static_out.hpp"

namespace merian_nodes {

// See corresponsing output.
template <typename ValueType>
class SpecialStaticIn : public TypedInputConnector<SpecialStaticOutHandle<ValueType>, void> {
  public:
    SpecialStaticIn(const std::string& name)
        : TypedInputConnector<SpecialStaticOutHandle<ValueType>, void>(name, 0) {}

    void resource([[maybe_unused]] const GraphResourceHandle& resource) override {}

    void on_connect_output(const OutputConnectorHandle& output) override {
        auto casted_output = std::dynamic_pointer_cast<SpecialStaticOut<ValueType>>(output);
        if (!casted_output) {
            throw graph_errors::connector_error{fmt::format(
                "SpecialStaticIn {} cannot recive from {}.", Connector::name, output->name)};
        }
    }
};

} // namespace merian_nodes
