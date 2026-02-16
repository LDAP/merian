#include "merian-nodes/graph/graph_description.hpp"
#include "merian/utils/properties_json_load.hpp"

#include <regex>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace merian_nodes {

// -----------------------------------------------------------------
// Node and Connection Management
// -----------------------------------------------------------------

const std::string& GraphDescription::add_node(const std::string& node_type,
                                              const std::optional<std::string>& identifier,
                                              const nlohmann::json& config) {
    std::string node_id = identifier.value_or(generate_unique_identifier(node_type));

    validate_identifier(node_id, "Node identifier");

    if (nodes.contains(node_id)) {
        throw std::invalid_argument(
            fmt::format("Node with identifier '{}' already exists", node_id));
    }

    PerNodeInfo info;
    info.node_type = node_type;
    info.config = config;
    nodes[node_id] = std::move(info);

    hash++;                            // Structure changed
    return nodes.find(node_id)->first; // Return the key from the map
}

bool GraphDescription::remove_node(const std::string& identifier) {
    if (!nodes.contains(identifier)) {
        return false;
    }

    // Remove all connections to/from this node
    // First, remove outgoing connections from other nodes that target this node
    for (auto& [node_id, node_info] : nodes) {
        // Remove from incoming connections
        for (auto it = node_info.incoming_connections.begin();
             it != node_info.incoming_connections.end();) {
            if (it->second.first == identifier) { // src_node matches
                it = node_info.incoming_connections.erase(it);
            } else {
                ++it;
            }
        }

        // Remove from outgoing connections
        for (auto& [output_name, output_info] : node_info.outgoing_connections) {
            for (auto it = output_info.target.begin(); it != output_info.target.end();) {
                if (it->first == identifier) { // dst_node matches
                    it = output_info.target.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    nodes.erase(identifier);
    hash++; // Structure changed
    return true;
}

void GraphDescription::add_connection(const std::string& src,
                                      const std::string& dst,
                                      const std::string& src_output,
                                      const std::string& dst_input) {
    // Add to outgoing connections of source node
    if (nodes.contains(src)) {
        auto& src_node = nodes.at(src);
        src_node.outgoing_connections[src_output].target[dst].insert(dst_input);
    }

    // Add to incoming connections of destination node
    if (nodes.contains(dst)) {
        auto& dst_node = nodes.at(dst);
        dst_node.incoming_connections[dst_input] = {src, src_output};
    }

    hash++; // Structure changed
}

bool GraphDescription::remove_connection(const std::string& src,
                                         const std::string& dst,
                                         const std::string& dst_input) {
    bool removed = false;

    // Find src_output from incoming connections
    std::string src_output;
    if (nodes.contains(dst)) {
        auto& dst_node = nodes.at(dst);
        auto it = dst_node.incoming_connections.find(dst_input);
        if (it != dst_node.incoming_connections.end() && it->second.first == src) {
            src_output = it->second.second;
            dst_node.incoming_connections.erase(it);
            removed = true;
        }
    }

    if (removed) {
        // Remove from source node's outgoing connections
        assert(nodes.contains(src));
        auto& src_node = nodes.at(src);
        assert(src_node.outgoing_connections.contains(src_output));
        auto& output_info = src_node.outgoing_connections.at(src_output);
        assert(output_info.target.contains(dst));
        output_info.target.at(dst).erase(dst_input);
        // If no more inputs to this dst, remove the dst entry
        if (output_info.target.at(dst).empty()) {
            output_info.target.erase(dst);
        }

        hash++; // Structure changed
    }

    return removed;
}

// -----------------------------------------------------------------
// Serialization
// -----------------------------------------------------------------

GraphDescription GraphDescription::from_json(const nlohmann::json& json) {
    GraphDescription description;

    if (!json.contains(SCHEMA_VERSION_KEY)) {
        parse_graph_v1(json, description);
        return description;
    }

    int schema_version = json[SCHEMA_VERSION_KEY].get<int>();
    if (schema_version == 2) {
        parse_graph_v2(json, description);
    } else if (schema_version == 3) {
        parse_graph_v3(json, description);
    } else {
        throw std::runtime_error{fmt::format("schema version {} unsupported.", schema_version)};
    }

    return description;
}

nlohmann::json GraphDescription::to_json() const {
    nlohmann::json json;
    dump_graph_v3(json);
    return json;
}

void GraphDescription::dump_graph_v2(nlohmann::json& json) const {
    json[SCHEMA_VERSION_KEY] = 2;

    if (!graph_properties.empty()) {
        json["graph_properties"] = graph_properties;
    }

    if (!profiler_properties.empty()) {
        json["profiler"] = profiler_properties;
    }

    nlohmann::json nodes_json;
    for (const auto& [identifier, node_info] : nodes) {
        nlohmann::json node_json;
        node_json["type"] = node_info.node_type;
        node_json["disable"] = !node_info.enabled;

        if (!node_info.config.empty()) {
            node_json["properties"] = node_info.config;
        }

        nodes_json[identifier] = node_json;
    }
    if (!nodes_json.empty()) {
        json["nodes"] = nodes_json;
    }

    nlohmann::json connections_json = nlohmann::json::array();
    for (const auto& [src, node_info] : nodes) {
        for (const auto& [src_output, output_info] : node_info.outgoing_connections) {
            for (const auto& [dst, dst_inputs] : output_info.target) {
                for (const auto& dst_input : dst_inputs) {
                    nlohmann::json connection;
                    connection["src"] = src;
                    connection["src_output"] = src_output;
                    connection["dst"] = dst;
                    connection["dst_input"] = dst_input;
                    connections_json.push_back(connection);
                }
            }
        }
    }
    if (!connections_json.empty()) {
        json["connections"] = connections_json;
    }
}

void GraphDescription::dump_graph_v3(nlohmann::json& json) const {
    json[SCHEMA_VERSION_KEY] = 3;

    if (!graph_properties.empty()) {
        json["graph_properties"] = graph_properties;
    }

    if (!profiler_properties.empty()) {
        json["profiler"] = profiler_properties;
    }

    nlohmann::json nodes_json = nlohmann::json::array();
    for (const auto& [identifier, node_info] : nodes) {
        nlohmann::json node_json;
        node_json["id"] = identifier;
        node_json["type"] = node_info.node_type;
        node_json["enabled"] = node_info.enabled;

        if (!node_info.config.empty()) {
            node_json["properties"] = node_info.config;
        }

        if (!node_info.metadata.empty()) {
            node_json["metadata"] = node_info.metadata;
        }

        if (!node_info.outgoing_connections.empty()) {
            nlohmann::json outputs_array = nlohmann::json::array();
            for (const auto& [output_name, output_info] : node_info.outgoing_connections) {
                for (const auto& [dst, dst_inputs] : output_info.target) {
                    for (const auto& dst_input : dst_inputs) {
                        outputs_array.push_back(
                            fmt::format("{}->{}.{}", output_name, dst, dst_input));
                    }
                }
            }
            node_json["outputs"] = outputs_array;
        }

        nodes_json.push_back(node_json);
    }
    if (!nodes_json.empty()) {
        json["nodes"] = nodes_json;
    }
}

void GraphDescription::parse_graph_v1(const nlohmann::json& json, GraphDescription& description) {
    // Parse graph properties
    if (json.contains("graph_properties")) {
        description.graph_properties = json["graph_properties"];
    }

    // Parse profiler properties
    if (json.contains("profiler")) {
        description.profiler_properties = json["profiler"];
    }

    // Parse nodes
    if (json.contains("nodes")) {
        const auto& nodes_json = json["nodes"];
        for (auto it = nodes_json.begin(); it != nodes_json.end(); ++it) {
            const std::string& identifier = it.key();
            const auto& node_json = it.value();

            std::string node_type = node_json["type"].get<std::string>();
            nlohmann::json config =
                node_json.contains("properties") ? node_json["properties"] : nlohmann::json{};
            bool disabled =
                node_json.contains("disable") ? node_json["disable"].get<bool>() : false;

            description.add_node(node_type, identifier, config);
            description.set_node_enabled(identifier, !disabled);
        }
    }

    // Parse connections
    if (json.contains("connections")) {
        const auto& connections_json = json["connections"];
        for (const auto& connection : connections_json) {
            std::string src = connection["src"].get<std::string>();
            std::string dst = connection["dst"].get<std::string>();
            std::string src_output = connection["src_output"].get<std::string>();
            std::string dst_input = connection["dst_input"].get<std::string>();

            description.add_connection(src, dst, src_output, dst_input);
        }
    }
}

void GraphDescription::parse_graph_v2(const nlohmann::json& json, GraphDescription& description) {
    // V2 format is the same as V1, just with schema_version field
    parse_graph_v1(json, description);
}

void GraphDescription::parse_graph_v3(const nlohmann::json& json, GraphDescription& description) {
    if (json.contains("graph_properties")) {
        description.graph_properties = json["graph_properties"];
    }

    if (json.contains("profiler")) {
        description.profiler_properties = json["profiler"];
    }

    if (json.contains("nodes")) {
        const auto& nodes_json = json["nodes"];
        for (const auto& node_json : nodes_json) {
            std::string identifier = node_json["id"].get<std::string>();
            std::string node_type = node_json["type"].get<std::string>();
            nlohmann::json config =
                node_json.contains("properties") ? node_json["properties"] : nlohmann::json{};
            bool enabled = node_json.contains("enabled") ? node_json["enabled"].get<bool>() : true;

            description.add_node(node_type, identifier, config);
            description.set_node_enabled(identifier, enabled);

            if (node_json.contains("metadata")) {
                description.set_node_metadata(identifier, node_json["metadata"]);
            }

            if (node_json.contains("outputs")) {
                const auto& outputs_json = node_json["outputs"];

                if (outputs_json.is_object()) {
                    // Format: {"output_name": ["dst.input", ...]}
                    for (auto it = outputs_json.begin(); it != outputs_json.end(); ++it) {
                        const std::string& output_name = it.key();
                        const auto& targets_array = it.value();

                        for (const auto& target_str : targets_array) {
                            auto [dst, dst_input] =
                                parse_dot_target(target_str.get<std::string>(), identifier, output_name);
                            description.add_connection(identifier, dst, output_name, dst_input);
                        }
                    }
                } else if (outputs_json.is_array()) {
                    // Format: ["output->dst.input", ...]
                    for (const auto& conn_str : outputs_json) {
                        auto [output_name, dst, dst_input] =
                            parse_arrow_connection(conn_str.get<std::string>(), identifier);
                        description.add_connection(identifier, dst, output_name, dst_input);
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------
// Helper Methods
// -----------------------------------------------------------------

void GraphDescription::validate_identifier(const std::string& identifier,
                                            const std::string& context) {
    if (identifier.find('.') != std::string::npos) {
        throw std::invalid_argument(
            fmt::format("{} '{}' contains reserved character '.'", context, identifier));
    }
    if (identifier.find("->") != std::string::npos) {
        throw std::invalid_argument(
            fmt::format("{} '{}' contains reserved sequence '->'", context, identifier));
    }
    if (identifier.find('/') != std::string::npos) {
        throw std::invalid_argument(
            fmt::format("{} '{}' contains reserved character '/'", context, identifier));
    }
}

std::tuple<std::string, std::string, std::string>
GraphDescription::parse_arrow_connection(const std::string& connection,
                                         const std::string& node_id) {
    static const std::regex conn_regex(R"(^([^-]+)->([^.]+)\.(.+)$)");
    std::smatch match;
    if (std::regex_match(connection, match, conn_regex)) {
        return {match[1], match[2], match[3]};
    }
    throw std::runtime_error(
        fmt::format("Invalid connection '{}' in node '{}': expected 'output->dst.input'",
                    connection, node_id));
}

std::pair<std::string, std::string>
GraphDescription::parse_dot_target(const std::string& target, const std::string& node_id,
                                   const std::string& output_name) {
    static const std::regex target_regex(R"(^([^.]+)\.(.+)$)");
    std::smatch match;
    if (std::regex_match(target, match, target_regex)) {
        return {match[1], match[2]};
    }
    throw std::runtime_error(fmt::format(
        "Invalid connection target '{}' for output '{}' in node '{}': expected 'dst.input'", target,
        output_name, node_id));
}

std::string GraphDescription::generate_unique_identifier(const std::string& node_type) {
    std::string base_id = node_type;
    std::string candidate = base_id;
    int counter = 0;

    while (nodes.contains(candidate)) {
        candidate = fmt::format("{} {}", base_id, counter);
        counter++;
    }

    return candidate;
}

} // namespace merian_nodes
