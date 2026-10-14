#include "merian-graph/graph/node_io.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian-graph/graph/graph.hpp"
#include "merian-graph/graph/graph_data.hpp"

#include "merian/shader/shader_object.hpp"

namespace merian {

// --- NodeIOLayout ---

bool NodeIOLayout::input_optional(const InputConnectorHandle& input_connector) const {
    return data->input_descriptors.at(input_connector).optional;
}

bool NodeIOLayout::is_connected(const OutputConnectorHandle& output_connector) const {
    return !data->output_connections.at(output_connector).inputs.empty();
}

OutputConnectorHandle
NodeIOLayout::output_for_input(const InputConnectorHandle& input_connector) const {
#ifndef NDEBUG
    const auto input_name = [&] {
        return data->input_descriptors.contains(input_connector)
                   ? data->input_descriptors.at(input_connector).name
                   : "<unknown>";
    };
    if (!allow_delayed && data->input_descriptors.at(input_connector).delay > 0) {
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

const ContextHandle& NodeIOLayout::get_context() const {
    return graph->context;
}

const ResourceAllocatorHandle& NodeIOLayout::get_allocator() const {
    return data->allocator;
}

// --- NodeIO ---

bool NodeIO::input_optional(const InputConnectorHandle& input_connector) const {
    return data->input_descriptors.at(input_connector).optional;
}

const GraphResourceHandle&
NodeIO::resource_for_input(const InputConnectorHandle& input_connector) const {
    return std::get<0>(data->input_connections.at(input_connector).precomputed_resources[set_idx]);
}

const GraphResourceHandle&
NodeIO::connected_resource_for_input(const InputConnectorHandle& input_connector) const {
    const GraphResourceHandle& resource = resource_for_input(input_connector);
    if (!resource) {
        throw graph_errors::node_error{fmt::format(
            "input {} is not connected", data->input_descriptors.at(input_connector).name)};
    }
    return resource;
}

const GraphResourceHandle&
NodeIO::resource_for_output(const OutputConnectorHandle& output_connector) const {
    const GraphResourceHandle& resource =
        std::get<0>(data->output_connections.at(output_connector).precomputed_resources[set_idx]);
    const OutputConnectorDescriptor& descriptor = data->output_descriptors.at(output_connector);
    if (!resource) {
        throw graph_errors::node_error{
            fmt::format("output {} is disabled and has no resource", descriptor.name)};
    }
    if (!allow_transient_outputs && !descriptor.persistent) {
        throw graph_errors::node_error{
            fmt::format("output {} is accessed in on_connected but not persistent: its contents do "
                        "not survive until the node runs",
                        descriptor.name)};
    }
    return resource;
}

bool NodeIO::is_connected(const OutputConnectorHandle& output_connector) const {
    return !data->output_connections.at(output_connector).inputs.empty();
}

const ContextHandle& NodeIO::get_context() const {
    return graph->context;
}

const ResourceAllocatorHandle& NodeIO::get_allocator() const {
    return data->allocator;
}

std::any& NodeIO::get_frame_data() const {
    return graph->ring_fences.get().user_data.in_flight_data[node];
}

void NodeIO::bind(const ShaderObjectHandle& object) const {
    bind(object->get_cursor());
}

void NodeIO::bind(ShaderCursor cursor) const {
    ShaderCursor graph_in = cursor.find("graph_in");
    ShaderCursor graph_out = cursor.find("graph_out");

    const auto bind_connector = [&](const ConnectorHandle& connector, ShaderCursor& group,
                                    const GraphResourceHandle& resource,
                                    const ConnectorAccess& access, const bool transient_output) {
        if (!connector->shader_bindable()) {
            return;
        }
        const std::string& name = data->bind_field_name.at(connector);
        ShaderCursor field = cursor.find(name);
        if (!field.is_valid()) {
            field = group.find(name);
        }
        if (!field.is_valid()) {
            return;
        }
        if (resource && transient_output && !allow_transient_outputs) {
            throw graph_errors::node_error{
                fmt::format("{} is bound in on_connected but not persistent: its contents do not "
                            "survive until the node runs",
                            name)};
        }
        connector->bind(field, resource, graph->resource_allocator, access);
    };

    for (const auto& [input, per_input] : data->input_connections) {
        // null for unconnected optional inputs - the connector writes a dummy then
        const GraphResourceHandle resource =
            per_input.node ? std::get<0>(per_input.precomputed_resources[set_idx]) : nullptr;
        bind_connector(input, graph_in, resource, data->input_descriptors.at(input).access, false);
    }
    for (const auto& [output, per_output] : data->output_connections) {
        // the node binds what it writes in place as its input
        if (per_output.version_of) {
            continue;
        }
        const OutputConnectorDescriptor& descriptor = data->output_descriptors.at(output);
        bind_connector(output, graph_out, std::get<0>(per_output.precomputed_resources[set_idx]),
                       descriptor.access, !descriptor.persistent);
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
