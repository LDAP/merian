#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/graph/resource.hpp"

namespace merian_nodes {

template <typename ValueType> class SpecialStaticOut;
template <typename ValueType>
using SpecialStaticOutHandle = std::shared_ptr<SpecialStaticOut<ValueType>>;

// Stores a static value in the output connector.
// Makes it possible to access it directly in the describe_outputs() of a reciving node and enforces
// a graph rebuild if the value has to change.
template <typename ValueType>
class SpecialStaticOut : public TypedOutputConnector<const ValueType&>, public GraphResource {
  public:
    SpecialStaticOut(const std::string& name, const ValueType& value)
        : TypedOutputConnector<const ValueType&>(name, false), connector_value(value) {}

    using ResourceType = SpecialStaticOut<ValueType>;

    virtual GraphResourceHandle create_resource(
        [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
        [[maybe_unused]] const ResourceAllocatorHandle& allocator,
        [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
        [[maybe_unused]] const uint32_t resoruce_index,
        [[maybe_unused]] const uint32_t ring_size) override {

        if (has_new) {
            connector_value = std::move(new_connector_value);
            has_new = false;
        }

        return std::static_pointer_cast<ResourceType>(Connector::shared_from_this());
    }

    const ValueType& resource([[maybe_unused]] const GraphResourceHandle& resource) override {
        return connector_value;
    }

    Connector::ConnectorStatusFlags on_post_process(
        [[maybe_unused]] GraphRun& run,
        [[maybe_unused]] const vk::CommandBuffer& cmd,
        [[maybe_unused]] GraphResourceHandle& resource,
        [[maybe_unused]] const NodeHandle& node,
        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override {

        if (has_new) {
            return Connector::NEEDS_RECONNECT;
        }

        return {};
    }

    void properties(Properties& config) override {
        if constexpr (fmt::is_formattable<ValueType>()) {
            config.output_text(fmt::format("Current value: {}", connector_value));
        }
    }

    const ValueType& value() const {
        return connector_value;
    }

    // Setting the value results in a graph rebuild
    void operator=(const ValueType& new_value) {
        set(new_value);
    }

    // Calling this method results in a graph rebuild
    void set(const ValueType& new_value) {
        new_connector_value = new_value;
        has_new = true;
    }

  public:
    static SpecialStaticOutHandle<ValueType> create(const std::string& name,
                                                    const ValueType& value) {
        return std::make_shared<SpecialStaticOut<ValueType>>(name, value);
    }

  public:
    ValueType connector_value;

  private:
    ValueType new_connector_value;
    bool has_new = false;
};

} // namespace merian_nodes
