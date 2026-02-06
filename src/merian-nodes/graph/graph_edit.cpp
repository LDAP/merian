#include "merian-nodes/graph/graph.hpp"

#include <spdlog/spdlog.h>

namespace merian {

const std::string& Graph::add_node(const std::string& node_name,
                                   const std::optional<std::string>& identifier) {

    if (!identifier) {
        // Preserve node name if we can. add_node below uses the node type.
        std::string node_identifier;
        uint32_t i = 0;
        do {
            node_identifier = fmt::format("{} {}", node_name, i++);
        } while (node_for_identifier.contains(node_identifier));

        return add_node(NodeRegistry::get_instance().create_node_from_name(node_name), node_identifier);
    }

    return add_node(NodeRegistry::get_instance().create_node_from_name(node_name), identifier);
}

NodeHandle Graph::find_node_for_identifier(const std::string& identifier) const {
    if (!node_for_identifier.contains(identifier)) {
        return nullptr;
    }
    return node_for_identifier.at(identifier);
}

void Graph::add_connection(const std::string& src,
                           const std::string& dst,
                           const std::string& src_output,
                           const std::string& dst_input) {
    const NodeHandle src_node = find_node_for_identifier(src);
    const NodeHandle dst_node = find_node_for_identifier(dst);
    assert(src_node);
    assert(dst_node);
    add_connection(src_node, dst_node, src_output, dst_input);
}

bool Graph::remove_connection(const std::string& src,
                              const std::string& dst,
                              const std::string& dst_input) {
    const NodeHandle src_node = find_node_for_identifier(src);
    const NodeHandle dst_node = find_node_for_identifier(dst);
    assert(src_node);
    assert(dst_node);
    return remove_connection(src_node, dst_node, dst_input);
}

bool Graph::remove_node(const std::string& identifier) {
    if (!node_for_identifier.contains(identifier)) {
        return false;
    }

    std::function<void()> remove_task = [this, identifier] {
        wait();

        const NodeHandle node = node_for_identifier.at(identifier);
        const NodeData& data = node_data.at(node);

        for (auto it = data.desired_outgoing_connections.begin();
             it != data.desired_outgoing_connections.end();
             it = data.desired_outgoing_connections.begin()) {
            remove_connection(node, it->dst, it->dst_input);
        }

        for (auto it = data.desired_incoming_connections.begin();
             it != data.desired_incoming_connections.end();
             it = data.desired_incoming_connections.begin()) {
            remove_connection(it->second.first, node, it->first);
        }

        const std::string node_identifier = data.identifier;
        node_data.erase(node);
        node_for_identifier.erase(identifier);
        for (uint32_t i = 0; i < ring_fences.size(); i++) {
            InFlightData& in_flight_data = ring_fences.get(i).user_data;
            in_flight_data.in_flight_data.erase(node);
        }

        SPDLOG_DEBUG("removed node {} ({})", node_identifier, NodeRegistry::get_instance().node_type_name(node));
        needs_reconnect = true;
    };

    if (run_in_progress) {
        SPDLOG_DEBUG("schedule removal of node {} for the end of run the current run.", identifier);
        on_run_finished_tasks.emplace_back(std::move(remove_task));
    } else {
        remove_task();
    }

    return true;
}

const std::string& Graph::add_node(const std::shared_ptr<Node>& node,
                                   const std::optional<std::string>& identifier) {
    if (node_data.contains(node)) {
        throw std::invalid_argument{
            fmt::format("graph already contains this node as '{}'", node_data.at(node).identifier)};
    }

    std::string node_identifier;
    if (identifier) {
        if (identifier->empty()) {
            throw std::invalid_argument{"node identifier cannot be empty"};
        }
        if (node_for_identifier.contains(identifier.value())) {
            throw std::invalid_argument{fmt::format(
                "graph already contains a node with identifier '{}'", identifier.value())};
        }
        if (*identifier == "user") {
            throw std::invalid_argument{"the identifier 'user' is reserved"};
        }
        if (*identifier == "graph") {
            throw std::invalid_argument{"the identifier 'graph' is reserved"};
        }
        node_identifier = identifier.value();
    } else {
        uint32_t i = 0;
        do {
            node_identifier = fmt::format("{} {}", NodeRegistry::get_instance().node_type_name(node), i++);
        } while (node_for_identifier.contains(node_identifier));
    }

    node_for_identifier[node_identifier] = node;
    auto [it, inserted] = node_data.try_emplace(node, node_identifier);
    assert(inserted);

    node->initialize(context, resource_allocator);

    needs_reconnect = true;
    SPDLOG_DEBUG("added node {} ({})", node_identifier, NodeRegistry::get_instance().node_type_name(node));

    return it->second.identifier;
}

void Graph::add_connection(const NodeHandle& src,
                           const NodeHandle& dst,
                           const std::string& src_output,
                           const std::string& dst_input) {
    if (!node_data.contains(src) || !node_data.contains(dst)) {
        throw std::invalid_argument{"graph does not contain the source or destination node"};
    }

    NodeData& src_data = node_data.at(src);
    NodeData& dst_data = node_data.at(dst);

    if (dst_data.desired_incoming_connections.contains(dst_input)) {
        const auto& [old_src, old_src_output] = dst_data.desired_incoming_connections.at(dst_input);
        [[maybe_unused]] const NodeData& old_src_data = node_data.at(old_src);
        SPDLOG_DEBUG("remove conflicting connection {}, {} ({}) -> {}, {} ({})", old_src_output,
                     old_src_data.identifier, NodeRegistry::get_instance().node_type_name(old_src), dst_input,
                     dst_data.identifier, NodeRegistry::get_instance().node_type_name(dst));
        remove_connection(old_src, dst, dst_input);
    }

    {
        // outgoing
        [[maybe_unused]] const auto [it, inserted] =
            src_data.desired_outgoing_connections.emplace(dst, src_output, dst_input);
        assert(inserted);
    }

    {
        // incoming
        [[maybe_unused]] const auto [it, inserted] =
            dst_data.desired_incoming_connections.try_emplace(dst_input, src, src_output);
        assert(inserted);
    }

    needs_reconnect = true;
    SPDLOG_DEBUG("added connection {}, {} ({}) -> {}, {} ({})", src_output, src_data.identifier,
                 NodeRegistry::get_instance().node_type_name(src), dst_input, dst_data.identifier,
                 NodeRegistry::get_instance().node_type_name(dst));
}

bool Graph::remove_connection(const NodeHandle src,
                              const NodeHandle dst,
                              const std::string dst_input) {
    // Developer note: Pass by reference is not used because this might be called with
    // references to iterator objects of the sets/maps we edit.

    if (!node_data.contains(src) || !node_data.contains(dst)) {
        throw std::invalid_argument{"graph does not contain the source or destination node"};
    }
    NodeData& src_data = node_data.at(src);
    NodeData& dst_data = node_data.at(dst);

    const auto it = dst_data.desired_incoming_connections.find(dst_input);
    if (it == dst_data.desired_incoming_connections.end()) {
        SPDLOG_WARN("connection {} ({}) -> {}, {} ({}) does not exist and cannot be removed.",
                    src_data.identifier, NodeRegistry::get_instance().node_type_name(src), dst_input,
                    dst_data.identifier, NodeRegistry::get_instance().node_type_name(dst));
        return false;
    }

    const std::string src_output = it->second.second;
    dst_data.desired_incoming_connections.erase(it);

    const auto out_it = src_data.desired_outgoing_connections.find({dst, src_output, dst_input});
    // else we did not add the connection properly
    assert(out_it != src_data.desired_outgoing_connections.end());
    src_data.desired_outgoing_connections.erase(out_it);
    SPDLOG_DEBUG("removed connection {}, {} ({}) -> {}, {} ({})", src_output, src_data.identifier,
                 NodeRegistry::get_instance().node_type_name(src), dst_input, dst_data.identifier,
                 NodeRegistry::get_instance().node_type_name(dst));

    needs_reconnect = true;
    return true;

    // Note: Since the connections are not needed in a graph run we do not need to wait until
    // the end of a run to remove the conenction.
}

} // namespace merian
