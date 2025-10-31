#pragma once

#include "connector_input.hpp"

#include <any>

namespace merian {

class GraphEvent {
  public:
    struct Info {
        // nullptr sent by the user or graph
        const NodeHandle& sender_node;
        // "" sent by the user or graph
        const std::string& node_name;
        // "user" if sent by the user or "graph" if sent by runtime
        const std::string& identifier;

        const std::string& event_name;
    };

    using Data = std::any;

    using Listener = std::function<bool(const GraphEvent::Info&, const GraphEvent::Data&)>;
};

// Access the outputs that are connected to your inputs.
class NodeIOLayout {
  public:
    NodeIOLayout(const std::function<OutputConnectorHandle(const InputConnectorHandle&)>& io_layout,
                 const std::function<void(const std::string&, const GraphEvent::Listener&)>&
                     register_event_listener_f)
        : io_layout(io_layout), register_event_listener_f(register_event_listener_f) {}

    // Behavior undefined if an optional input connector is not connected.
    template <
        typename T,
        typename OutputConnectorType = T::output_connector_type,
        std::enable_if_t<std::is_base_of_v<OutputAccessibleInputConnector<OutputConnectorType>, T>,
                         bool> = true>
    OutputConnectorType operator[](const std::shared_ptr<T>& input_connector) const {
        assert(io_layout(input_connector) && "optional input connector is not connected");
        return input_connector->output_connector(io_layout(input_connector));
    }

    // Returns if an input is connected. This is always true for non-optional inputs.
    bool is_connected(const InputConnectorHandle& input_connector) const {
        // if not optional, an output must exist!
        assert(input_connector->optional || io_layout(input_connector));
        return io_layout(input_connector) != nullptr;
    }

    /*
     * Event pattern:
     * - nodeType/nodeIdentifier/eventName
     * - /user/eventName (user events, sent using the graph methods)
     * - /graph/eventName (user events, sent using the graph methods)
     * - comma separated list of those patterns
     *
     * Empty nodeType, nodeIdentifier, eventName mean "any".
     *
     * The listener recives info about the event and optional data. The listener can return if the
     * event was processed. In this case processing ends if "notify_all = false", otherwise the
     * event is distributed to all listeners.
     */
    void register_event_listener(const std::string& event_pattern,
                                 const GraphEvent::Listener& event_listerner) const {
        register_event_listener_f(event_pattern, event_listerner);
    }

  private:
    const std::function<OutputConnectorHandle(const InputConnectorHandle&)> io_layout;

    const std::function<void(const std::string&, const GraphEvent::Listener&)>
        register_event_listener_f;
};

class NodeIO {
  public:
    NodeIO(const std::function<GraphResourceHandle(const InputConnectorHandle&)>&
               resource_for_input_connector,
           const std::function<GraphResourceHandle(const OutputConnectorHandle&)>&
               resource_for_output_connector,
           const std::function<bool(const OutputConnectorHandle&)>& output_is_connected,
           const std::function<std::any&()>& get_frame_data,
           const std::function<void(
               const std::string& event_name, const GraphEvent::Data&, const bool)>& send_event_f,
           const std::function<uint32_t(const InputConnectorHandle&)> binding_for_input_connector,
           const std::function<uint32_t(const OutputConnectorHandle&)> binding_for_output_connector)
        : resource_for_input_connector(resource_for_input_connector),
          resource_for_output_connector(resource_for_output_connector),
          output_is_connected(output_is_connected), get_frame_data(get_frame_data),
          send_event_f(send_event_f), binding_for_input_connector(binding_for_input_connector),
          binding_for_output_connector(binding_for_output_connector) {}

    // Behavior undefined if an optional input connector is not connected.
    template <typename T,
              typename ResourceAccessType = T::resource_access_type,
              std::enable_if_t<std::is_base_of_v<AccessibleConnector<ResourceAccessType>, T> &&
                                   std::is_base_of_v<InputConnector, T>,
                               bool> = true>
    ResourceAccessType operator[](const std::shared_ptr<T>& input_connector) const {
        assert(input_connector && "input connector cannot be null");
        assert((input_connector->optional || resource_for_input_connector(input_connector)) &&
               "non-optional input connector is not connected. This should be prevented by the "
               "Graph.");
        assert((!input_connector->optional || resource_for_input_connector(input_connector)) &&
               "optional input connector is not connected");
        return input_connector->resource(resource_for_input_connector(input_connector));
    }

    template <typename T,
              typename ResourceAccessType = T::resource_access_type,
              std::enable_if_t<std::is_base_of_v<AccessibleConnector<ResourceAccessType>, T> &&
                                   std::is_base_of_v<OutputConnector, T>,
                               bool> = true>
    ResourceAccessType operator[](const std::shared_ptr<T>& output_connector) const {
        return output_connector->resource(resource_for_output_connector(output_connector));
    }

    // Returns if an input is connected. This is always true for non-optional inputs.
    bool is_connected(const InputConnectorHandle& input_connector) const {
        // if not optional, a resource must exist!
        assert(input_connector->optional || resource_for_input_connector(input_connector));
        return resource_for_input_connector(input_connector) != nullptr;
    }

    // Returns if at least one input is connected to this output.
    bool is_connected(const OutputConnectorHandle& output_connector) const {
        return output_is_connected(output_connector);
    }

    // Returns a reference to the frame data as the template type.
    //
    // If no frame data exists it will constructed with the given parameters, if the cast fails the
    // frame data will be replaced with the new type.
    template <typename T, typename... Args> T& frame_data(Args&&... args) const {
        std::any& data = get_frame_data();
        if (!data.has_value()) {
            data = std::make_any<T>(std::forward<Args>(args)...);
        }

        return std::any_cast<T&>(data);
    }

    void send_event(const std::string& event_name,
                    const GraphEvent::Data& data = {},
                    const bool notify_all = true) const {
        send_event_f(event_name, data, notify_all);
    }

    uint32_t get_binding(const InputConnectorHandle& input_connector) const {
        const uint32_t binding = binding_for_input_connector(input_connector);
        assert(binding != DescriptorSet::NO_DESCRIPTOR_BINDING);
        return binding;
    }

    uint32_t get_binding(const OutputConnectorHandle& output_connector) const {
        const uint32_t binding = binding_for_output_connector(output_connector);
        assert(binding != DescriptorSet::NO_DESCRIPTOR_BINDING);
        return binding;
    }

  private:
    const std::function<GraphResourceHandle(const InputConnectorHandle&)>
        resource_for_input_connector;
    const std::function<GraphResourceHandle(const OutputConnectorHandle&)>
        resource_for_output_connector;

    const std::function<bool(const OutputConnectorHandle&)> output_is_connected;

    const std::function<std::any&()> get_frame_data;

    const std::function<void(const std::string& event_name, const GraphEvent::Data&, const bool)>
        send_event_f;

    const std::function<uint32_t(const InputConnectorHandle&)> binding_for_input_connector;

    const std::function<uint32_t(const OutputConnectorHandle&)> binding_for_output_connector;
};
} // namespace merian
