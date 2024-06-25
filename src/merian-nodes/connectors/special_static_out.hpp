#include "merian-nodes/graph/connector_output.hpp"

namespace merian_nodes {

// Stores a static value in the output connector.
// Makes it possible to access it directly in the describe_outputs() of a reciving node and enforces
// a graph rebuild if the value has to change.
template <typename ValueType> class SpecialStaticOut : public TypedOutputConnector<void> {
  public:
    SpecialStaticOut(const std::string& name, const ValueType& value)
        : TypedOutputConnector(name, true), value(value) {}

    virtual GraphResourceHandle create_resource(
        [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
        [[maybe_unused]] const ResourceAllocatorHandle& allocator,
        [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
        [[maybe_unused]] const uint32_t resoruce_index,
        [[maybe_unused]] const uint32_t ring_size) override {

        // empty resource.
        return std::make_shared<GraphResource>();
    }

    void resource([[maybe_unused]] const GraphResourceHandle& resource) override {}

    void properties(Properties& config) override {
        if constexpr (fmt::is_formattable<ValueType>()) {
            config.output_text(fmt::format("Current value: {}", value));
        }
    }

  public:
    const ValueType value;
};

template <typename ValueType>
using SpecialStaticOutHandle = std::shared_ptr<SpecialStaticOut<ValueType>>;

} // namespace merian_nodes
