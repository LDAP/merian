#pragma once

#include "connector_input.hpp"

#include <any>

namespace merian {

class Graph;
class Node;
using NodeHandle = std::shared_ptr<Node>;

namespace graph_internal {
struct NodeData;
}

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
    NodeIOLayout(Graph* graph,
                 graph_internal::NodeData* data,
                 const NodeHandle& node,
                 const bool allow_delayed)
        : graph(graph), data(data), node(node), allow_delayed(allow_delayed) {}

    // Behavior undefined if an optional input connector is not connected.
    template <
        typename T,
        typename OutputConnectorType = T::output_connector_type,
        std::enable_if_t<std::is_base_of_v<OutputAccessibleInputConnector<OutputConnectorType>, T>,
                         bool> = true>
    OutputConnectorType operator[](const std::shared_ptr<T>& input_connector) const {
        assert(output_for_input(input_connector) && "optional input connector is not connected");
        return input_connector->output_connector(output_for_input(input_connector));
    }

    // Returns if an input is connected. This is always true for non-optional inputs.
    bool is_connected(const InputConnectorHandle& input_connector) const {
        // if not optional, an output must exist!
        assert(input_optional(input_connector) || output_for_input(input_connector));
        return output_for_input(input_connector) != nullptr;
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
                                 const GraphEvent::Listener& event_listener) const;

  private:
    OutputConnectorHandle output_for_input(const InputConnectorHandle& input_connector) const;
    bool input_optional(const InputConnectorHandle& input_connector) const;

    Graph* graph;
    graph_internal::NodeData* data;
    NodeHandle node;
    bool allow_delayed;
};

class NodeIO {
  public:
    NodeIO(Graph* graph,
           graph_internal::NodeData* data,
           const NodeHandle& node,
           const uint32_t set_idx)
        : graph(graph), data(data), node(node), set_idx(set_idx) {}

    // Behavior undefined if an optional input connector is not connected.
    template <typename T,
              typename ResourceAccessType = T::resource_access_type,
              std::enable_if_t<std::is_base_of_v<AccessibleConnector<ResourceAccessType>, T> &&
                                   std::is_base_of_v<InputConnector, T>,
                               bool> = true>
    ResourceAccessType operator[](const std::shared_ptr<T>& input_connector) const {
        assert(input_connector && "input connector cannot be null");
        assert((input_optional(input_connector) || resource_for_input(input_connector)) &&
               "non-optional input connector is not connected. This should be prevented by the "
               "Graph.");
        assert((!input_optional(input_connector) || resource_for_input(input_connector)) &&
               "optional input connector is not connected");
        return input_connector->resource(resource_for_input(input_connector));
    }

    template <typename T,
              typename ResourceAccessType = T::resource_access_type,
              std::enable_if_t<std::is_base_of_v<AccessibleConnector<ResourceAccessType>, T> &&
                                   std::is_base_of_v<OutputConnector, T>,
                               bool> = true>
    ResourceAccessType operator[](const std::shared_ptr<T>& output_connector) const {
        return output_connector->resource(resource_for_output(output_connector));
    }

    // Returns if an input is connected. This is always true for non-optional inputs.
    bool is_connected(const InputConnectorHandle& input_connector) const {
        // if not optional, a resource must exist!
        assert(input_optional(input_connector) || resource_for_input(input_connector));
        return resource_for_input(input_connector) != nullptr;
    }

    // Returns if at least one input is connected to this output.
    bool is_connected(const OutputConnectorHandle& output_connector) const;

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

    // Binds every shader-visible input and output whose name matches a cursor field:
    // input "src" -> "in_src", output "irr" -> "out_irr", either at the cursor root or inside
    // a "graph_in" / "graph_out" struct field. Unmatched ports are skipped (the shader may not
    // need them); unconnected optional inputs with a matching field receive a dummy.
    void bind(const ShaderObjectHandle& object) const;
    void bind(ShaderCursor cursor) const;

    void send_event(const std::string& event_name,
                    const GraphEvent::Data& event_data = {},
                    const bool notify_all = true) const;

  private:
    GraphResourceHandle resource_for_input(const InputConnectorHandle& input_connector) const;
    GraphResourceHandle resource_for_output(const OutputConnectorHandle& output_connector) const;
    bool input_optional(const InputConnectorHandle& input_connector) const;
    std::any& get_frame_data() const;

    Graph* graph;
    graph_internal::NodeData* data;
    NodeHandle node;
    uint32_t set_idx;
};

} // namespace merian
