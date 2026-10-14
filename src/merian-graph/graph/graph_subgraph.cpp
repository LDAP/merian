#include "merian-graph/graph/graph.hpp"
#include "merian-graph/graph/graph_description.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <unordered_set>

namespace merian {

namespace {
// (node name, connector name) of "node.connector"; the node name is empty for a public port.
std::optional<std::pair<std::string, std::string>> split_port(const std::string& port) {
    const size_t dot = port.find('.');
    if (dot == std::string::npos || dot + 1 == port.size()) {
        return std::nullopt;
    }
    return std::make_pair(port.substr(0, dot), port.substr(dot + 1));
}

std::optional<std::string> subgraph_declaration_error(const NodeHandle& node,
                                                      const Subgraph& subgraph) {
    std::unordered_set<std::string> names;
    std::unordered_set<NodeHandle> children;
    for (const auto& [name, child] : subgraph.get_nodes()) {
        try {
            GraphDescription::validate_identifier(name, "child name");
        } catch (const std::invalid_argument& e) {
            return e.what();
        }
        if (!names.insert(name).second) {
            return fmt::format("child name '{}' is taken", name);
        }
        if (!child || !children.insert(child).second || child == node) {
            return fmt::format("child {} is null, its parent or added twice", name);
        }
        try {
            NodeRegistry::get_instance().node_type(child);
        } catch (const std::invalid_argument& e) {
            return fmt::format("child {}: {}", name, e.what());
        }
    }

    std::unordered_set<std::string> connected_inputs;
    std::unordered_set<std::string> public_inputs;
    std::unordered_set<std::string> public_outputs;
    for (const auto& [src, dst] : subgraph.get_connections()) {
        const auto src_port = split_port(src);
        const auto dst_port = split_port(dst);
        if (!src_port || !dst_port ||
            (!src_port->first.empty() && !names.contains(src_port->first)) ||
            (!dst_port->first.empty() && !names.contains(dst_port->first))) {
            return fmt::format("connection {} -> {} names an unknown port", src, dst);
        }
        if (src_port->first.empty() && dst_port->first.empty()) {
            return fmt::format("connection {} -> {} connects two public ports", src, dst);
        }
        if (src_port->first.empty()) {
            public_inputs.insert(src_port->second);
        }
        if (dst_port->first.empty() ? !public_outputs.insert(dst_port->second).second
                                    : !connected_inputs.insert(dst).second) {
            return fmt::format("{} is connected twice", dst);
        }
    }

    std::unordered_set<std::string> passed_on;
    for (const auto& [input, output] : subgraph.get_pass_throughs()) {
        const auto input_port = split_port(input);
        const auto output_port = split_port(output);
        if (!input_port || !output_port || !input_port->first.empty() ||
            !output_port->first.empty() || !public_inputs.contains(input_port->second) ||
            !public_outputs.contains(output_port->second) ||
            !passed_on.insert(output_port->second).second) {
            return fmt::format("invalid pass-through {} -> {}", input, output);
        }
    }
    return std::nullopt;
}
} // namespace

void Graph::build_subgraphs() {
    // 1. the subgraphs of the nodes in the graph, so that children can move between parents
    std::unordered_map<NodeHandle, std::optional<Subgraph>> subgraphs;
    std::unordered_map<NodeHandle, std::vector<NodeHandle>> describers;
    const auto describe = [&](const NodeHandle& node) {
        const auto [it, inserted] = subgraphs.emplace(node, subgraph_of(node));
        if (it->second) {
            for (const auto& [name, child] : it->second->get_nodes()) {
                describers[child].push_back(node);
            }
        }
    };
    std::vector<NodeHandle> nodes = top_level_nodes();
    for (size_t i = 0; i < nodes.size(); i++) {
        describe(nodes[i]);
        const std::vector<NodeHandle>& children = node_data.at(nodes[i]).children;
        nodes.insert(nodes.end(), children.begin(), children.end());
    }

    // 2. parents before their children, those added on the way described when reached
    nodes = top_level_nodes();
    for (size_t i = 0; i < nodes.size(); i++) {
        if (!subgraphs.contains(nodes[i])) {
            describe(nodes[i]);
        }
        build_subgraph(nodes[i], subgraphs.at(nodes[i]), describers);
        const std::vector<NodeHandle>& children = node_data.at(nodes[i]).children;
        nodes.insert(nodes.end(), children.begin(), children.end());
    }

    // 3. children that left their parent for one that left the graph
    std::vector<NodeHandle> orphans;
    for (const auto& [node, data] : node_data) {
        if (!data.parent) {
            continue;
        }
        const auto parent = node_data.find(data.parent);
        if (parent == node_data.end() ||
            std::ranges::find(parent->second.children, node) == parent->second.children.end()) {
            orphans.push_back(node);
        }
    }
    for (const NodeHandle& orphan : orphans) {
        if (node_data.contains(orphan)) {
            erase_node(orphan);
        }
    }
}

std::optional<Subgraph> Graph::subgraph_of(const NodeHandle& node) {
    NodeData& data = node_data.at(node);
    // not initialized
    if (data.unsupported) {
        return std::nullopt;
    }
    std::optional<Subgraph> subgraph;
    collect_node_errors(data.errors, [&] {
        subgraph = node->describe_subgraph();
        if (const std::optional<std::string> error = subgraph_declaration_error(node, *subgraph)) {
            data.errors.emplace_back(*error);
        }
    });
    if (data.errors.empty()) {
        return subgraph;
    }
    SPDLOG_ERROR("{} failed to describe its subgraph and is disabled:\n - {}", data.label,
                 fmt::join(data.errors, "\n - "));
    return std::nullopt;
}

void Graph::build_subgraph(
    const NodeHandle& node,
    const std::optional<Subgraph>& subgraph,
    const std::unordered_map<NodeHandle, std::vector<NodeHandle>>& describers) {
    // the children of a node that is unsupported or failed stay
    if (!subgraph) {
        return;
    }
    NodeData& data = node_data.at(node);
    const auto described_by = [&](const NodeHandle& child) -> const std::vector<NodeHandle>& {
        static const std::vector<NodeHandle> none;
        const auto it = describers.find(child);
        return it == describers.end() ? none : it->second;
    };
    // a parent keeps a child it describes; one whose subgraph failed keeps all
    for (const auto& [name, child] : subgraph->get_nodes()) {
        const auto it = node_data.find(child);
        if (it == node_data.end() || it->second.parent == node) {
            continue;
        }
        const NodeHandle& owner = it->second.parent;
        if (!owner || !node_data.at(owner).errors.empty() || node_data.at(owner).unsupported ||
            std::ranges::find(described_by(child), owner) != described_by(child).end()) {
            data.errors.emplace_back(
                fmt::format("child {} is already in the graph as {}", name, it->second.identifier));
            SPDLOG_ERROR("{}: {}", data.label, data.errors.back());
            return;
        }
    }

    // 1. children: removed, taken from other parents, renamed with their children, added
    const std::unordered_map<std::string, NodeHandle> child_for_name(subgraph->get_nodes().begin(),
                                                                     subgraph->get_nodes().end());
    std::vector<NodeHandle> children;
    std::ranges::transform(subgraph->get_nodes(), std::back_inserter(children),
                           [](const auto& entry) { return entry.second; });
    std::erase_if(data.children, [&](const NodeHandle& child) {
        if (std::ranges::find(children, child) != children.end()) {
            return false;
        }
        if (std::ranges::none_of(described_by(child),
                                 [&](const NodeHandle& parent) { return parent != node; })) {
            erase_node(child);
        }
        return true;
    });
    std::vector<std::pair<NodeHandle, std::string>> renamed;
    for (const auto& [name, child] : subgraph->get_nodes()) {
        const auto it = node_data.find(child);
        if (it == node_data.end()) {
            continue;
        }
        if (it->second.parent != node) {
            if (const auto parent = node_data.find(it->second.parent); parent != node_data.end()) {
                std::erase(parent->second.children, child);
            }
            it->second.parent = node;
        }
        if (const std::string identifier = fmt::format("{}:{}", data.identifier, name);
            it->second.identifier != identifier) {
            renamed.emplace_back(child, identifier);
        }
    }
    for (size_t i = 0; i < renamed.size(); i++) {
        const auto [renamed_node, identifier] = renamed[i];
        const NodeData& renamed_data = node_data.at(renamed_node);
        for (const NodeHandle& child : renamed_data.children) {
            renamed.emplace_back(child, identifier + node_data.at(child).identifier.substr(
                                                         renamed_data.identifier.size()));
        }
        // a new child may have taken the name already
        if (const auto it = node_for_identifier.find(renamed_data.identifier);
            it != node_for_identifier.end() && it->second == renamed_node) {
            node_for_identifier.erase(it);
        }
    }
    for (const auto& [renamed_node, identifier] : renamed) {
        node_data.at(renamed_node).set_identifier(identifier);
        node_for_identifier[identifier] = renamed_node;
    }
    for (const auto& [name, child] : subgraph->get_nodes()) {
        if (!node_data.contains(child)) {
            // listed before inserting, which can throw
            data.children.push_back(child);
            insert_node(child, fmt::format("{}:{}", data.identifier, name), node);
        }
    }
    data.children = std::move(children);

    // 2. public ports and the connections between the children
    for (const auto& [src, dst] : subgraph->get_connections()) {
        const auto [src_name, src_port] = *split_port(src);
        const auto [dst_name, dst_port] = *split_port(dst);
        if (src_name.empty()) {
            data.public_inputs[src_port].push_back({child_for_name.at(dst_name), dst_port});
        } else if (dst_name.empty()) {
            data.public_outputs.try_emplace(dst_port, child_for_name.at(src_name), src_port);
        } else {
            data.subgraph_connections.push_back(
                {child_for_name.at(src_name), src_port, child_for_name.at(dst_name), dst_port});
        }
    }
    for (const auto& [input, output] : subgraph->get_pass_throughs()) {
        data.public_pass_throughs.try_emplace(split_port(output)->second,
                                              split_port(input)->second);
    }
}

void Graph::connect_subgraphs() {
    std::vector<NodeHandle> nodes = top_level_nodes();
    for (size_t i = 0; i < nodes.size(); i++) {
        NodeData& data = node_data.at(nodes[i]);
        connect_subgraph(data);
        nodes.insert(nodes.end(), data.children.begin(), data.children.end());
    }
}

void Graph::connect_subgraph(NodeData& data) {
    if (data.children.empty()) {
        return;
    }
    const auto fail = [&](const std::string& error) {
        SPDLOG_ERROR("{}: {}", data.label, error);
        data.errors.emplace_back(error);
    };

    // 1. the inputs it connects exist, unless the child or its subgraph is unknown
    const auto receives = [&](const Port& input) {
        const NodeData& child_data = node_data.at(input.node);
        return !child_data.errors.empty() || child_data.unsupported ||
               child_data.input_connector_for_name.contains(input.name) ||
               child_data.public_inputs.contains(input.name);
    };
    std::vector<Port> connected_inputs;
    for (const Connection& connection : data.subgraph_connections) {
        connected_inputs.push_back({connection.dst, connection.dst_input});
    }
    for (const auto& [port, inputs] : data.public_inputs) {
        connected_inputs.insert(connected_inputs.end(), inputs.begin(), inputs.end());
    }
    for (const Port& input : connected_inputs) {
        if (data.errors.empty() && !receives(input)) {
            fail(fmt::format("its subgraph connects input {} of {}, which does not exist",
                             input.name, node_data.at(input.node).identifier));
        }
    }
    for (const auto& [output, input] : data.public_pass_throughs) {
        if (data.errors.empty() && data.pass_through_inputs.contains(output)) {
            fail(fmt::format("two inputs pass through to {}", output));
        }
    }
    for (const auto& [output, source] : data.public_outputs) {
        if (const auto input = data.input_connector_for_name.find(output);
            data.errors.empty() && input != data.input_connector_for_name.end() &&
            data.input_descriptors.at(input->second).in_place) {
            fail(fmt::format("its input {} is written in place and named like a public output",
                             output));
        }
    }
    if (!data.errors.empty()) {
        data.subgraph_connections.clear();
        data.public_inputs.clear();
        data.public_outputs.clear();
        data.public_pass_throughs.clear();
        return;
    }
    data.pass_through_inputs.insert(data.public_pass_throughs.begin(),
                                    data.public_pass_throughs.end());

    // 2. what feeds the children's inputs: the subgraph, else their auto connection
    for (const Connection& connection : data.subgraph_connections) {
        node_data.at(connection.dst)
            .subgraph_sources.try_emplace(connection.dst_input, connection.src,
                                          connection.src_output);
    }
    const NodeHandle& parent = node_for_identifier.at(data.identifier);
    for (const auto& [port, inputs] : data.public_inputs) {
        // an unwired public input leaves the children's auto connections in place
        const bool wired = wiring(parent, port).has_value();
        for (const Port& input : inputs) {
            NodeData& child_data = node_data.at(input.node);
            const auto connector = child_data.input_connector_for_name.find(input.name);
            if (wired || connector == child_data.input_connector_for_name.end() ||
                !child_data.input_descriptors.at(connector->second).auto_connect) {
                child_data.public_feeds.try_emplace(input.name, port);
            }
        }
    }
    for (const NodeHandle& child : data.children) {
        NodeData& child_data = node_data.at(child);
        // (node, output), or (null, public input)
        const auto source_of = [&](const std::string& input) -> std::optional<Port> {
            if (const auto source = child_data.subgraph_sources.find(input);
                source != child_data.subgraph_sources.end()) {
                return source->second;
            }
            if (const auto port = child_data.public_feeds.find(input);
                port != child_data.public_feeds.end()) {
                return Port{nullptr, port->second};
            }
            return std::nullopt;
        };
        for (const InputConnectorHandle& input : child_data.input_connectors) {
            const InputConnectorDescriptor& descriptor = child_data.input_descriptors.at(input);
            if (!descriptor.auto_connect || source_of(descriptor.name)) {
                continue;
            }
            const std::optional<Port> source =
                auto_connection_source(child, child_data, descriptor.name, source_of);
            if (source && source->node) {
                child_data.subgraph_sources.try_emplace(descriptor.name, *source);
            } else if (source) {
                child_data.public_feeds.try_emplace(descriptor.name, source->name);
            }
        }
    }
}

} // namespace merian
