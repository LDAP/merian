#include "merian-graph/graph/graph_description.hpp"

#include <algorithm>
#include <regex>
#include <stdexcept>
#include <tuple>

namespace merian {

// -----------------------------------------------------------------
// Node and Connection Management
// -----------------------------------------------------------------

const std::string& GraphDescription::add_node(const std::string& node_type,
                                              const std::optional<std::string>& identifier,
                                              const nlohmann::json& config) {
    const std::string node_id =
        identifier ? *identifier : unique_identifier(node_type, [&](const std::string& candidate) {
            return nodes.contains(candidate);
        });

    validate_identifier(node_id, "Node identifier");

    if (nodes.contains(node_id)) {
        throw std::invalid_argument(
            fmt::format("Node with identifier '{}' already exists", node_id));
    }

    PerNodeInfo info;
    info.node_type = node_type;
    info.config = config;
    const auto [it, inserted] = nodes.emplace(node_id, std::move(info));
    assert(inserted);

    return it->first;
}

bool GraphDescription::remove_node(const std::string& identifier) {
    if (!nodes.contains(identifier)) {
        return false;
    }

    for (auto& [node_id, node_info] : nodes) {
        std::erase_if(node_info.incoming_connections,
                      [&identifier](const auto& conn) { return conn.second.first == identifier; });
    }

    nodes.erase(identifier);
    return true;
}

void GraphDescription::add_connection(const std::string& src,
                                      const std::string& dst,
                                      const std::string& src_output,
                                      const std::string& dst_input) {
    const auto src_node_it = nodes.find(src);
    const auto dst_node_it = nodes.find(dst);
    for (const auto& [identifier, it] :
         {std::pair{src, src_node_it}, std::pair{dst, dst_node_it}}) {
        if (it == nodes.end()) {
            throw std::invalid_argument(fmt::format("connection {}.{} -> {}.{}: no node {}", src,
                                                    src_output, dst, dst_input, identifier));
        }
    }
    dst_node_it->second.incoming_connections.insert_or_assign(dst_input,
                                                              std::pair{src, src_output});
}

bool GraphDescription::remove_connection(const std::string& src,
                                         const std::string& dst,
                                         const std::string& dst_input) {
    const auto dst_node_it = nodes.find(dst);
    if (dst_node_it == nodes.end()) {
        return false;
    }
    auto& incoming = dst_node_it->second.incoming_connections;
    const auto it = incoming.find(dst_input);
    if (it == incoming.end() || it->second.first != src) {
        return false;
    }
    incoming.erase(it);
    return true;
}

// -----------------------------------------------------------------
// Serialization
// -----------------------------------------------------------------

GraphDescription GraphDescription::from_json(const nlohmann::json& json) {
    GraphDescription description;
    switch (const int version = json.value(SCHEMA_VERSION_KEY, SCHEMA_VERSION)) {
    case 4:
        parse_graph_v4(json, description);
        break;
    default:
        throw std::runtime_error{fmt::format("schema version {} unsupported.", version)};
    }
    return description;
}

nlohmann::json GraphDescription::to_json() const {
    nlohmann::json json;
    dump_graph_v4(json);
    return json;
}

void GraphDescription::dump_graph_v4(nlohmann::json& json) const {
    json[SCHEMA_VERSION_KEY] = 4;

    if (!graph_properties.empty()) {
        json["graph_properties"] = graph_properties;
    }

    if (!profiler_properties.empty()) {
        json["profiler"] = profiler_properties;
    }

    if (!cli.empty()) {
        json["cli"] = cli;
    }

    // (src -> (src output, dst, dst input))
    std::map<std::string, std::vector<std::tuple<std::string, std::string, std::string>>> outputs;
    for (const auto& [dst, node_info] : nodes) {
        for (const auto& [dst_input, source] : node_info.incoming_connections) {
            outputs[source.first].emplace_back(source.second, dst, dst_input);
        }
    }

    nlohmann::json nodes_json = nlohmann::json::object();
    for (const auto& [identifier, node_info] : nodes) {
        nlohmann::json node_json;
        node_json["type"] = node_info.node_type;
        node_json["enabled"] = node_info.enabled;
        if (node_info.linearization_order != 0) {
            node_json["linearization order"] = node_info.linearization_order;
        }

        if (!node_info.config.empty()) {
            node_json["properties"] = node_info.config;
        }

        if (!node_info.metadata.empty()) {
            node_json["metadata"] = node_info.metadata;
        }

        if (const auto it = outputs.find(identifier); it != outputs.end()) {
            std::ranges::sort(it->second);
            nlohmann::json outputs_array = nlohmann::json::array();
            for (const auto& [output_name, dst, dst_input] : it->second) {
                outputs_array.push_back(fmt::format("{}->{}.{}", output_name, dst, dst_input));
            }
            node_json["outputs"] = outputs_array;
        }

        nodes_json[identifier] = node_json;
    }
    if (!nodes_json.empty()) {
        json["nodes"] = nodes_json;
    }
}

void GraphDescription::parse_graph_v4(const nlohmann::json& json, GraphDescription& description) {
    if (json.contains("graph_properties")) {
        description.graph_properties = json["graph_properties"];
    }

    if (json.contains("profiler")) {
        description.profiler_properties = json["profiler"];
    }

    if (json.contains("cli")) {
        description.cli = json["cli"];
    }

    if (json.contains("nodes")) {
        const auto& nodes_json = json["nodes"];
        // Nodes with Metadata
        for (const auto& [identifier, node_json] : nodes_json.items()) {
            if (!node_json.contains("type")) {
                throw std::runtime_error{fmt::format(
                    "node '{}' has no type (setting a property of a node the config does not "
                    "declare creates such an entry)",
                    identifier)};
            }
            const std::string node_type = node_json["type"].get<std::string>();
            const nlohmann::json config =
                node_json.contains("properties") ? node_json["properties"] : nlohmann::json{};
            const bool enabled =
                node_json.contains("enabled") ? node_json["enabled"].get<bool>() : true;

            description.add_node(node_type, identifier, config);
            description.set_node_enabled(identifier, enabled);
            if (node_json.contains("linearization order")) {
                description.set_node_linearization_order(
                    identifier, node_json["linearization order"].get<uint32_t>());
            }

            if (node_json.contains("metadata")) {
                description.set_node_metadata(identifier, node_json["metadata"]);
            }
        }
        // Connections
        for (const auto& [identifier, node_json] : nodes_json.items()) {
            if (node_json.contains("outputs")) {
                const auto& outputs_json = node_json["outputs"];

                if (outputs_json.is_object()) {
                    // Format: {"output_name": ["dst.input", ...]}
                    for (auto it = outputs_json.begin(); it != outputs_json.end(); ++it) {
                        const std::string& output_name = it.key();
                        const auto& targets_array = it.value();

                        for (const auto& target_str : targets_array) {
                            auto [dst, dst_input] = parse_dot_target(target_str.get<std::string>(),
                                                                     identifier, output_name);
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
    if (identifier.empty()) {
        throw std::invalid_argument(fmt::format("{} cannot be empty", context));
    }
    if (identifier == "user" || identifier == "graph") {
        throw std::invalid_argument(fmt::format("{} '{}' is reserved", context, identifier));
    }
    for (const std::string_view reserved : {".", "->", "/", ":"}) {
        if (identifier.find(reserved) != std::string::npos) {
            throw std::invalid_argument(
                fmt::format("{} '{}' contains reserved '{}'", context, identifier, reserved));
        }
    }
}

std::tuple<std::string, std::string, std::string>
GraphDescription::parse_arrow_connection(const std::string& connection,
                                         const std::string& node_id) {
    static const std::regex conn_regex(R"(^(.+?)->([^.]+)\.(.+)$)");
    std::smatch match;
    if (std::regex_match(connection, match, conn_regex)) {
        return {match[1], match[2], match[3]};
    }
    throw std::runtime_error(fmt::format(
        "Invalid connection '{}' in node '{}': expected 'output->dst.input'", connection, node_id));
}

std::pair<std::string, std::string> GraphDescription::parse_dot_target(
    const std::string& target, const std::string& node_id, const std::string& output_name) {
    static const std::regex target_regex(R"(^([^.]+)\.(.+)$)");
    std::smatch match;
    if (std::regex_match(target, match, target_regex)) {
        return {match[1], match[2]};
    }
    throw std::runtime_error(fmt::format(
        "Invalid connection target '{}' for output '{}' in node '{}': expected 'dst.input'", target,
        output_name, node_id));
}

std::string
GraphDescription::unique_identifier(const std::string& base,
                                    const std::function<bool(const std::string&)>& taken) {
    for (uint32_t i = 0;; i++) {
        std::string identifier = fmt::format("{} {}", base, i);
        if (!taken(identifier)) {
            return identifier;
        }
    }
}

// -----------------------------------------------------------------
// Config files
// -----------------------------------------------------------------

std::vector<std::filesystem::path> GraphDescription::default_search_dirs() {
    std::vector<std::filesystem::path> dirs;
    for (const char* name : {"res", "examples"}) {
        if (const auto dir = FileLoader::search_cwd_parents(name)) {
            dirs.push_back(*dir);
        }
    }
    if (const auto prefix = FileLoader::install_prefix()) {
        dirs.push_back(*prefix / FileLoader::install_datadir_name() / "merian" / "examples");
    }
    return dirs;
}

std::filesystem::path
GraphDescription::find_config(const std::filesystem::path& path,
                              const std::vector<std::filesystem::path>& search_dirs) {
    if (std::filesystem::exists(path)) {
        return path;
    }
    if (const auto found = FileLoader(search_dirs).find_file(path)) {
        return *found;
    }
    throw graph_errors::config_error{{fmt::format("config '{}' not found", path.string())}};
}

void GraphDescription::merge_into(nlohmann::json& base, const nlohmann::json& overwrite) {
    if (!overwrite.is_object()) {
        base = overwrite;
        return;
    }
    if (!base.is_object()) {
        base = nlohmann::json::object();
    }

    static constexpr std::string_view append_prefix = "$+$";
    for (const auto& [key, value] : overwrite.items()) {
        if (value.is_array() && key.starts_with(append_prefix)) {
            nlohmann::json& target = base[key.substr(append_prefix.size())];
            if (!target.is_array()) {
                target = nlohmann::json::array();
            }
            for (const auto& element : value) {
                if (std::ranges::find(target, element) == target.end()) {
                    target.push_back(element);
                }
            }
        } else {
            merge_into(base[key], value);
        }
    }
}

} // namespace merian
