#include "merian-graph/graph/graph.hpp"
#include "merian-graph/graph/graph_description.hpp"
#include "merian-graph/merian_graph_extension.hpp"

#include <spdlog/spdlog.h>

namespace merian {

const std::string& Graph::add_node(const std::string& node_name,
                                   const std::optional<std::string>& identifier) {
    return add_node(registry.create_node_from_name(node_name),
                    identifier ? *identifier
                               : GraphDescription::unique_identifier(
                                     node_name, [this](const std::string& candidate) {
                                         return node_for_identifier.contains(candidate);
                                     }));
}

NodeHandle Graph::find_node_for_identifier(const std::string& identifier) const {
    const auto it = node_for_identifier.find(identifier);
    return it == node_for_identifier.end() ? nullptr : it->second;
}

void Graph::add_connection(const std::string& src,
                           const std::string& dst,
                           const std::string& src_output,
                           const std::string& dst_input) {
    add_connection(node_for(src), node_for(dst), src_output, dst_input);
}

bool Graph::remove_connection(const std::string& src,
                              const std::string& dst,
                              const std::string& dst_input) {
    return remove_connection(node_for(src), node_for(dst), dst_input);
}

bool Graph::remove_node(const std::string& identifier) {
    const NodeHandle node = find_node_for_identifier(identifier);
    if (!node) {
        return false;
    }
    if (const NodeHandle& parent = node_data.at(node).parent) {
        SPDLOG_WARN("node {} belongs to the subgraph of {} and cannot be removed", identifier,
                    node_data.at(parent).identifier);
        return false;
    }
    when_idle([this, node] {
        if (node_data.contains(node)) {
            wait();
            erase_node(node);
            request_reconnect();
        }
    });
    return true;
}

void Graph::when_idle(std::function<void()> edit) {
    deferred_edits.push_back(std::move(edit));
    apply_deferred_edits();
}

void Graph::apply_deferred_edits() {
    while (busy == 0 && !deferred_edits.empty()) {
        const std::function<void()> edit = std::move(deferred_edits.front());
        deferred_edits.erase(deferred_edits.begin());
        edit();
    }
}

const std::string& Graph::add_node(const std::shared_ptr<Node>& node,
                                   const std::string& identifier) {
    if (node_data.contains(node)) {
        throw std::invalid_argument{
            fmt::format("graph already contains this node as '{}'", node_data.at(node).identifier)};
    }
    GraphDescription::validate_identifier(identifier, "Node identifier");
    if (node_for_identifier.contains(identifier)) {
        throw std::invalid_argument{
            fmt::format("graph already contains a node with identifier '{}'", identifier)};
    }

    insert_node(node, identifier);
    request_reconnect();
    return node_data.at(node).identifier;
}

NodeHandle Graph::node_for(const std::string& identifier) const {
    const NodeHandle node = find_node_for_identifier(identifier);
    if (!node) {
        throw std::invalid_argument{fmt::format("graph does not contain node '{}'", identifier)};
    }
    return node;
}

void Graph::insert_node(const NodeHandle& node,
                        const std::string& identifier,
                        const NodeHandle& parent) {
    node_for_identifier[identifier] = node;
    [[maybe_unused]] const auto [it, inserted] =
        node_data.try_emplace(node, identifier, registry.node_type_name(node));
    assert(inserted);
    NodeData& data = it->second;
    data.parent = parent;
    data.memory_tracker =
        TrackingMemoryAllocator::create(resource_allocator->get_memory_allocator());
    data.allocator = std::make_shared<ResourceAllocator>(resource_allocator, data.memory_tracker);

    const InstanceSupportInfo& instance_support = context_extension->get_instance_support(node);
    if (instance_support.supported) {
        const DeviceSupportInfo& device_support = context_extension->get_device_support(node);
        data.unsupported = !device_support.supported;
        data.unsupported_reason = device_support.unsupported_reason;
    } else {
        data.unsupported = true;
        data.unsupported_reason = instance_support.unsupported_reason;
    }

    if (!data.unsupported) {
        try {
            node->initialize(context, data.allocator);
        } catch (const std::exception& e) {
            data.unsupported = true;
            data.unsupported_reason = fmt::format("initialize failed: {}", e.what());
        }
    }
    if (data.unsupported) {
        SPDLOG_WARN("{} is unsupported: {}", data.label, data.unsupported_reason);
    }
}

void Graph::erase_node(const NodeHandle& node) {
    const NodeHandle removed = node;
    NodeData& data = node_data.at(node);
    for (const NodeHandle& child : data.children) {
        erase_node(child);
    }
    for (auto& [other, other_data] : node_data) {
        std::erase_if(other_data.desired_incoming_connections, [&](const auto& entry) {
            if (entry.second.node != node) {
                return false;
            }
            // an auto connection from this node connects again once there is a new source
            other_data.settled_auto_connections.erase(entry.first);
            return true;
        });
        std::erase_if(other_data.rejected_auto_connections,
                      [&](const auto& entry) { return entry.second.node == node; });
    }
    for (InFlightData& slot : in_flight) {
        slot.in_flight_data.erase(node);
    }
    std::erase_if(event_listeners,
                  [&](const EventListener& listener) { return listener.node == node; });
    if (const auto it = node_for_identifier.find(data.identifier);
        it != node_for_identifier.end() && it->second == node) {
        node_for_identifier.erase(it);
    }
    node_data.erase(removed);
    removed->on_removed();
}

std::vector<NodeHandle> Graph::top_level_nodes() const {
    std::vector<NodeHandle> nodes;
    for (const auto& [identifier, node] : node_for_identifier) {
        if (!node_data.at(node).parent) {
            nodes.push_back(node);
        }
    }
    return nodes;
}

void Graph::add_connection(const NodeHandle& src,
                           const NodeHandle& dst,
                           const std::string& src_output,
                           const std::string& dst_input) {
    for (const NodeHandle& node : {src, dst}) {
        if (const NodeHandle& parent = node_data.at(node).parent) {
            throw std::invalid_argument{
                fmt::format("node {} belongs to the subgraph of {} and cannot be connected",
                            node_data.at(node).identifier, node_data.at(parent).identifier)};
        }
    }
    NodeData& dst_data = node_data.at(dst);
    std::optional<ConnectorRef> replaced;
    if (const auto it = dst_data.desired_incoming_connections.find(dst_input);
        it != dst_data.desired_incoming_connections.end()) {
        replaced = it->second;
    }

    add_desired_connection(src, dst, src_output, dst_input);

    // inputs auto connected to what this one received move along unless they were rewired
    for (const auto& [input, descriptor] : dst_data.input_descriptors) {
        const auto following = dst_data.desired_incoming_connections.find(descriptor.name);
        if (replaced && *replaced != ConnectorRef{src, src_output} &&
            descriptor.auto_connect.from == AutoConnect::From::INPUT &&
            descriptor.auto_connect.name == dst_input &&
            following != dst_data.desired_incoming_connections.end() &&
            following->second == *replaced) {
            add_connection(src, dst, src_output, descriptor.name);
        }
    }
    request_reconnect();
}

void Graph::add_desired_connection(const NodeHandle& src,
                                   const NodeHandle& dst,
                                   const std::string& src_output,
                                   const std::string& dst_input) {
    node_data.at(dst).desired_incoming_connections.insert_or_assign(dst_input,
                                                                    ConnectorRef{src, src_output});
    SPDLOG_DEBUG("connected {}.{} -> {}.{}", node_data.at(src).identifier, src_output,
                 node_data.at(dst).identifier, dst_input);
}

bool Graph::remove_connection(const NodeHandle& src,
                              const NodeHandle& dst,
                              const std::string& dst_input) {
    NodeData& dst_data = node_data.at(dst);
    const auto it = dst_data.desired_incoming_connections.find(dst_input);
    if (it == dst_data.desired_incoming_connections.end() || it->second.node != src) {
        SPDLOG_WARN("{} -> {}.{} is not connected", node_data.at(src).identifier,
                    dst_data.identifier, dst_input);
        return false;
    }
    const ConnectorRef removed = it->second;
    dst_data.desired_incoming_connections.erase(it);

    // inputs auto connected to what this one received connect again once it is connected
    for (const auto& [input, descriptor] : dst_data.input_descriptors) {
        const auto following = dst_data.desired_incoming_connections.find(descriptor.name);
        if (descriptor.auto_connect.from == AutoConnect::From::INPUT &&
            descriptor.auto_connect.name == dst_input &&
            following != dst_data.desired_incoming_connections.end() &&
            following->second == removed) {
            remove_connection(removed.node, dst, descriptor.name);
            dst_data.settled_auto_connections.erase(descriptor.name);
        }
    }
    request_reconnect();
    return true;
}

} // namespace merian
