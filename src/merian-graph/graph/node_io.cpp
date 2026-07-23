#include "merian-graph/graph/node_io.hpp"

#include "merian-graph/graph/graph.hpp"
#include "merian-graph/graph/graph_data.hpp"

#include "merian/shader/shader_object.hpp"

namespace merian {

// --- NodeIOLayout ---

bool NodeIOLayout::input_optional(const InputConnectorHandle& input_connector) const {
    return data->input_optional.at(input_connector);
}

OutputConnectorHandle
NodeIOLayout::output_for_input(const InputConnectorHandle& input_connector) const {
#ifndef NDEBUG
    const auto input_name = [&] {
        return data->input_name_for_connector.contains(input_connector)
                   ? data->input_name_for_connector.at(input_connector)
                   : "<unknown>";
    };
    if (!allow_delayed && data->input_delay.at(input_connector) > 0) {
        throw std::runtime_error{
            fmt::format("Node {} tried to access an output connector that is connected "
                        "through a delayed input {} (which is not allowed here but only in "
                        "on_connected).",
                        graph->registry.node_type_name(node), input_name())};
    }
    if (std::ranges::find(data->input_connectors, input_connector) ==
        data->input_connectors.end()) {
        throw std::runtime_error{
            fmt::format("Node {} tried to get an output connector for an input {} "
                        "which was not returned in describe_inputs (which is not "
                        "how this works).",
                        graph->registry.node_type_name(node), input_name())};
    }
#endif
    // for optional inputs an input connection with nullptr output exists.
    return data->input_connections.at(input_connector).output;
}

void NodeIOLayout::register_event_listener(const std::string& event_pattern,
                                           const GraphEvent::Listener& event_listener) const {
    graph->register_event_listener_for_connect(event_pattern, event_listener);
}

// --- NodeIO ---

bool NodeIO::input_optional(const InputConnectorHandle& input_connector) const {
    return data->input_optional.at(input_connector);
}

GraphResourceHandle NodeIO::resource_for_input(const InputConnectorHandle& input_connector) const {
    // null if an optional input is not connected.
    return std::get<0>(data->input_connections.at(input_connector).precomputed_resources[set_idx]);
}

GraphResourceHandle
NodeIO::resource_for_output(const OutputConnectorHandle& output_connector) const {
    return std::get<0>(
        data->output_connections.at(output_connector).precomputed_resources[set_idx]);
}

bool NodeIO::is_connected(const OutputConnectorHandle& output_connector) const {
    return !data->output_connections.at(output_connector).inputs.empty();
}

std::any& NodeIO::get_frame_data() const {
    return graph->ring_fences.get().user_data.in_flight_data[node];
}

void NodeIO::bind(const ShaderObjectHandle& object) const {
    bind(object->get_cursor());
}

void NodeIO::bind(ShaderCursor cursor) const {
    ShaderCursor graph_in = cursor.has_field("graph_in") ? cursor["graph_in"] : ShaderCursor{};
    ShaderCursor graph_out = cursor.has_field("graph_out") ? cursor["graph_out"] : ShaderCursor{};

    const auto bind_connector = [&](const ConnectorHandle& connector, ShaderCursor& group,
                                    const GraphResourceHandle& resource) {
        if (!connector->shader_bindable()) {
            return;
        }
        const std::string& name = data->bind_field_name.at(connector);
        ShaderCursor field;
        if (cursor.has_field(name)) {
            field = cursor[name];
        } else if (group.is_valid() && group.has_field(name)) {
            field = group[name];
        } else {
            return; // the shader does not use this port
        }
        connector->bind(field, resource, graph->resource_allocator,
                        data->connector_access.at(connector));
    };

    for (const auto& [input, per_input] : data->input_connections) {
        // null for unconnected optional inputs - the connector writes a dummy then
        const GraphResourceHandle resource =
            per_input.node ? std::get<0>(per_input.precomputed_resources[set_idx]) : nullptr;
        bind_connector(input, graph_in, resource);
    }
    for (const auto& [output, per_output] : data->output_connections) {
        bind_connector(output, graph_out, std::get<0>(per_output.precomputed_resources[set_idx]));
    }
}

void NodeIO::send_event(const std::string& event_name,
                        const GraphEvent::Data& event_data,
                        const bool notify_all) const {
    graph->send_event(
        GraphEvent::Info{node, graph->registry.node_type_name(node), data->identifier, event_name},
        event_data, notify_all);
}

} // namespace merian
