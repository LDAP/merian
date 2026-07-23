#include "merian-graph/graph/graph_description.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace merian {

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
    const auto [it, inserted] = nodes.emplace(node_id, std::move(info));
    assert(inserted);

    hash++;           // Structure changed
    return it->first; // Return the key from the map
}

bool GraphDescription::remove_node(const std::string& identifier) {
    if (!nodes.contains(identifier)) {
        return false;
    }

    // Remove all connections to/from this node
    // First, remove outgoing connections from other nodes that target this node
    for (auto& [node_id, node_info] : nodes) {
        // Remove from incoming connections
        std::erase_if(node_info.incoming_connections,
                      [&identifier](const auto& conn) { return conn.second.first == identifier; });

        // Remove from outgoing connections
        for (auto& [output_name, output_info] : node_info.outgoing_connections) {
            std::erase_if(output_info.target,
                          [&identifier](const auto& target) { return target.first == identifier; });
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
    const auto src_node_it = nodes.find(src);
    assert(src_node_it != nodes.end());
    src_node_it->second.outgoing_connections[src_output].target[dst].insert(dst_input);

    // Add to incoming connections of destination node
    const auto dst_node_it = nodes.find(dst);
    assert(dst_node_it != nodes.end());
    dst_node_it->second.incoming_connections[dst_input] = {src, src_output};

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
    } else {
        const int schema_version = json[SCHEMA_VERSION_KEY].get<int>();
        if (schema_version == 2) {
            parse_graph_v2(json, description);
        } else if (schema_version == 3) {
            parse_graph_v3(json, description);
        } else if (schema_version == 4) {
            parse_graph_v4(json, description);
        } else {
            throw std::runtime_error{fmt::format("schema version {} unsupported.", schema_version)};
        }
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

    nlohmann::json nodes_json = nlohmann::json::object();
    for (const auto& [identifier, node_info] : nodes) {
        nlohmann::json node_json;
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

        nodes_json[identifier] = node_json;
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
        for (const auto& [identifier, node_json] : nodes_json.items()) {
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
    // V2 format is the same as V1, just with version field
    parse_graph_v1(json, description);
}

void GraphDescription::parse_graph_v3(const nlohmann::json& json, GraphDescription& description) {
    // V3 stores nodes as an array with an explicit "id"; re-key and reuse the V4 parser.
    nlohmann::json v4 = json;
    if (json.contains("nodes")) {
        nlohmann::json nodes_json = nlohmann::json::object();
        for (const auto& node_json : json["nodes"]) {
            nlohmann::json entry = node_json;
            const std::string identifier = entry.at("id").get<std::string>();
            entry.erase("id");
            nodes_json[identifier] = std::move(entry);
        }
        v4["nodes"] = std::move(nodes_json);
    }
    parse_graph_v4(v4, description);
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
                    "node '{}' has no type (a cli pointer to a nonexistent node id creates such "
                    "an entry - check the config's cli block)",
                    identifier)};
            }
            const std::string node_type = node_json["type"].get<std::string>();
            const nlohmann::json config =
                node_json.contains("properties") ? node_json["properties"] : nlohmann::json{};
            const bool enabled =
                node_json.contains("enabled") ? node_json["enabled"].get<bool>() : true;

            description.add_node(node_type, identifier, config);
            description.set_node_enabled(identifier, enabled);

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

// -----------------------------------------------------------------
// CLI overrides
// -----------------------------------------------------------------

namespace {

using json = nlohmann::json;
using JsonPointer = json::json_pointer;

struct Patch {
    std::vector<std::pair<JsonPointer, json>> assignments;
    std::vector<std::string> merges;
};

Patch parse_patch(const json& spec) {
    Patch patch;
    const json assignments = spec.value("set", json::object());
    for (const auto& [pointer, value] : assignments.items()) {
        patch.assignments.emplace_back(JsonPointer(pointer), value);
    }
    if (const auto it = spec.find("merge"); it != spec.end()) {
        if (it->is_array()) {
            for (const json& file : *it) {
                patch.merges.push_back(file.get<std::string>());
            }
        } else {
            patch.merges.push_back(it->get<std::string>());
        }
    }
    return patch;
}

std::optional<bool> parse_bool(std::string text) {
    std::ranges::transform(text, text.begin(), [](unsigned char c) { return std::tolower(c); });
    if (text.empty() || text == "on" || text == "true" || text == "1" || text == "yes") {
        return true;
    }
    if (text == "off" || text == "false" || text == "0" || text == "no") {
        return false;
    }
    return std::nullopt;
}

json typed_value(const json& config, const JsonPointer& pointer, const std::string& value) {
    if (config.contains(pointer) && config.at(pointer).is_string()) {
        return value;
    }
    const json parsed = json::parse(value, nullptr, false);
    return parsed.is_discarded() ? json(value) : parsed;
}

std::string join_keys(const std::map<std::string, Patch>& map, const char* separator) {
    std::string out;
    for (const auto& [key, _] : map) {
        out += out.empty() ? key : separator + key;
    }
    return out;
}

struct Override {
    enum class Kind : uint8_t { VALUE, FLAG, VARIANT };

    std::string name;
    Kind kind = Kind::VALUE;
    bool required = false;

    JsonPointer pointer;
    Patch extra;
    Patch on;
    Patch off;
    std::map<std::string, Patch> variants;
    std::optional<std::string> default_variant;
    // Persisted selection: set once a variant is applied, so a stored config neither reverts to
    // the default nor re-applies the variant's patch on rerun.
    std::optional<std::string> selected;
};

std::optional<Patch> resolve(const Override& o, const std::string& value, const json& config) {
    switch (o.kind) {
    case Override::Kind::VALUE: {
        Patch patch = o.extra;
        patch.assignments.insert(patch.assignments.begin(),
                                 {o.pointer, typed_value(config, o.pointer, value)});
        return patch;
    }
    case Override::Kind::FLAG:
        if (const auto enabled = parse_bool(value)) {
            return *enabled ? o.on : o.off;
        }
        SPDLOG_ERROR("--{} expects on|off, got '{}'", o.name, value);
        return std::nullopt;
    case Override::Kind::VARIANT:
        if (const auto it = o.variants.find(value); it != o.variants.end()) {
            return it->second;
        }
        SPDLOG_ERROR("--{}: unknown '{}', expected one of {}", o.name, value,
                     join_keys(o.variants, ", "));
        return std::nullopt;
    }
    return std::nullopt;
}

std::string usage(const Override& o) {
    switch (o.kind) {
    case Override::Kind::VALUE:
        return o.name == "--" ? "args... (after graph.json)" : fmt::format("--{} <value>", o.name);
    case Override::Kind::FLAG:
        return fmt::format("--{} <on|off>", o.name);
    case Override::Kind::VARIANT:
        return fmt::format("--{} <{}>", o.name, join_keys(o.variants, "|"));
    }
    return {};
}

std::string target(const Override& o) {
    switch (o.kind) {
    case Override::Kind::VALUE: {
        const std::string pointer = o.pointer.to_string();
        const size_t count = o.extra.assignments.size() + o.extra.merges.size();
        return count == 0 ? pointer : fmt::format("{} (+{} more)", pointer, count);
    }
    case Override::Kind::FLAG:
        return "on/off";
    case Override::Kind::VARIANT:
        if (o.selected) {
            return fmt::format("selected: {}", *o.selected);
        }
        return o.default_variant ? fmt::format("default: {}", *o.default_variant) : "";
    }
    return {};
}

Override::Kind classify(const json& spec) {
    if (const auto it = spec.find("type"); it != spec.end()) {
        const std::string type = it->get<std::string>();
        if (type == "variant") {
            return Override::Kind::VARIANT;
        }
        if (type == "flag") {
            return Override::Kind::FLAG;
        }
        return Override::Kind::VALUE;
    }
    if (spec.contains("variants")) {
        return Override::Kind::VARIANT;
    }
    return spec.contains("pointer") ? Override::Kind::VALUE : Override::Kind::FLAG;
}

std::vector<Override> parse_overrides(const json& config) {
    std::vector<Override> overrides;
    const json cli = config.value("cli", json::object());
    for (const auto& [name, spec] : cli.items()) {
        if (spec.is_null()) {
            continue; // a fragment removes an inherited override by merging null over it
        }
        Override entry;
        entry.name = name;
        entry.kind = classify(spec);
        entry.required = spec.value("required", false);
        switch (entry.kind) {
        case Override::Kind::VALUE:
            entry.pointer = JsonPointer(spec.at("pointer").get<std::string>());
            entry.extra = parse_patch(spec);
            break;
        case Override::Kind::FLAG:
            entry.on = parse_patch(spec);
            if (spec.contains("off")) {
                entry.off = parse_patch(spec.at("off"));
            }
            break;
        case Override::Kind::VARIANT: {
            const json variants = spec.value("variants", json::object());
            for (const auto& [variant, patch] : variants.items()) {
                entry.variants.emplace(variant, parse_patch(patch));
            }
            if (const auto it = spec.find("default"); it != spec.end()) {
                entry.default_variant = it->get<std::string>();
            }
            if (const auto it = spec.find("selected"); it != spec.end()) {
                entry.selected = it->get<std::string>();
            }
            break;
        }
        }
        overrides.push_back(std::move(entry));
    }
    return overrides;
}

std::optional<std::filesystem::path>
resolve_file(const std::string& file, const std::vector<std::filesystem::path>& search_dirs) {
    if (std::filesystem::path direct(file); std::filesystem::exists(direct)) {
        return direct;
    }
    return merian::FileLoader(search_dirs).find_file(file);
}

// Merges first so assignments can target (and win over) merged content.
bool apply_patch(json& config,
                 const Patch& patch,
                 const std::vector<std::filesystem::path>& search_dirs) {
    for (const std::string& file : patch.merges) {
        const auto path = resolve_file(file, search_dirs);
        if (!path) {
            SPDLOG_ERROR("merge file '{}' not found", file);
            return false;
        }
        std::ifstream stream(path->string());
        const json overwrite = json::parse(stream, nullptr, false);
        if (overwrite.is_discarded()) {
            SPDLOG_ERROR("merge file '{}' is not valid JSON", path->string());
            return false;
        }
        merian::GraphDescription::merge_into(config, overwrite);
    }
    for (const auto& [pointer, value] : patch.assignments) {
        config[pointer] = value;
    }
    return true;
}

// Applies pending default/selected variants until none remain; applied patches can merge
// fragments that declare further variants, hence the re-parse loop. A variant with a persisted
// "selected" counts as settled without re-applying its patch.
bool settle_variants(json& config,
                     const std::set<std::string>& suppressed,
                     const std::vector<std::filesystem::path>& search_dirs) {
    bool ok = true;
    std::set<std::string> settled;
    while (true) {
        const std::vector<Override> overrides = parse_overrides(config);
        const auto pending = std::ranges::find_if(overrides, [&](const Override& o) {
            return o.kind == Override::Kind::VARIANT && (o.default_variant || o.selected) &&
                   !settled.contains(o.name) && !suppressed.contains(o.name);
        });
        if (pending == overrides.end()) {
            return ok;
        }
        settled.insert(pending->name);
        if (pending->selected) {
            continue;
        }
        const auto variant = pending->variants.find(*pending->default_variant);
        if (variant == pending->variants.end()) {
            SPDLOG_ERROR("--{}: default '{}' is not declared", pending->name,
                         *pending->default_variant);
            ok = false;
            continue;
        }
        ok = apply_patch(config, variant->second, search_dirs) && ok;
        config["cli"][pending->name]["selected"] = *pending->default_variant;
    }
}

} // namespace

void GraphDescription::merge_into(nlohmann::json& base, const nlohmann::json& overwrite) {
    if (base.is_object() && overwrite.is_object()) {
        static constexpr std::string_view append_prefix = "$+$";
        for (const auto& [key, value] : overwrite.items()) {
            // "$+$key": [..] appends its (deduplicated) elements to base["key"] instead of replacing
            // it, so independent variant fragments can each contribute to the same array.
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
    } else {
        base = overwrite;
    }
}

bool GraphDescription::apply_cli(nlohmann::json& config,
                                 const std::vector<std::string>& cli_args,
                                 const std::vector<std::filesystem::path>& search_dirs) {
    // A raw "--name" presence suppresses that variant's default even if its value later fails.
    std::set<std::string> raw_names;
    for (const std::string& arg : cli_args) {
        if (arg.starts_with("--") && arg.size() > 2) {
            raw_names.insert(arg.substr(2));
        }
    }

    bool ok = settle_variants(config, raw_names, search_dirs);

    // Overrides declared by merged fragments must be recognized in the same invocation: consume
    // one recognized "--name value" at a time and re-parse after each application.
    std::vector<bool> consumed(cli_args.size(), false);
    std::set<std::string> given;
    while (true) {
        const std::vector<Override> overrides = parse_overrides(config);
        size_t applied = cli_args.size();
        for (size_t i = 0; i < cli_args.size(); i++) {
            const std::string& arg = cli_args[i];
            if (consumed[i] || !arg.starts_with("--") || arg.size() <= 2) {
                continue;
            }
            const std::string name = arg.substr(2);
            const bool declared =
                name == "merge" ||
                std::ranges::any_of(overrides, [&](const Override& o) { return o.name == name; });
            if (!declared) {
                continue;
            }
            if (i + 1 >= cli_args.size() || consumed[i + 1]) {
                SPDLOG_ERROR("missing value for override '{}'", arg);
                return false;
            }
            consumed[i] = consumed[i + 1] = true;
            given.insert(name);
            const std::string& value = cli_args[i + 1];
            if (name == "merge") {
                ok =
                    apply_patch(config, Patch{.assignments = {}, .merges = {value}}, search_dirs) &&
                    ok;
            } else {
                const auto it = std::ranges::find(overrides, name, &Override::name);
                const std::optional<Patch> patch = resolve(*it, value, config);
                ok = patch && apply_patch(config, *patch, search_dirs) && ok;
                if (patch && it->kind == Override::Kind::VARIANT) {
                    config["cli"][name]["selected"] = value;
                }
            }
            ok = settle_variants(config, raw_names, search_dirs) && ok;
            applied = i;
            break;
        }
        if (applied == cli_args.size()) {
            break;
        }
    }

    // Remaining tokens are positional and feed the "--" override, joined in order.
    std::string joined;
    bool positional = false;
    for (size_t i = 0; i < cli_args.size(); i++) {
        if (consumed[i]) {
            continue;
        }
        if (positional) {
            joined += ' ';
        }
        joined += cli_args[i];
        positional = true;
    }

    const std::vector<Override> overrides = parse_overrides(config);
    if (positional) {
        given.insert("--");
        const auto it = std::ranges::find(overrides, "--", &Override::name);
        if (it == overrides.end()) {
            SPDLOG_ERROR("positional arguments given but '--' is not declared in the cli block");
            ok = false;
        } else {
            const std::optional<Patch> patch = resolve(*it, joined, config);
            ok = patch && apply_patch(config, *patch, search_dirs) && ok;
        }
    }

    for (const Override& o : overrides) {
        if (!o.required || given.contains(o.name) || o.default_variant || o.selected) {
            continue;
        }
        // a value a stored session already materialized (e.g. the scene file) satisfies
        if (o.kind == Override::Kind::VALUE && config.contains(o.pointer)) {
            const json& value = config.at(o.pointer);
            if (value.is_string() ? !value.get<std::string>().empty() : !value.is_null()) {
                continue;
            }
        }
        if (o.name == "--") {
            SPDLOG_ERROR("missing required positional args (see --help)");
        } else {
            SPDLOG_ERROR("missing required --{}", o.name);
        }
        ok = false;
    }

    // Null entries only mark removal (see parse_overrides); drop them from the stored config.
    if (const auto cli = config.find("cli"); cli != config.end() && cli->is_object()) {
        for (auto it = cli->begin(); it != cli->end();) {
            it = it->is_null() ? cli->erase(it) : std::next(it);
        }
    }

    return ok;
}

std::string GraphDescription::cli_help(const nlohmann::json& config,
                                       const std::vector<std::filesystem::path>& search_dirs) {
    // Settle pending selections on a copy so overrides declared by merged fragments show up.
    nlohmann::json settled = config;
    settle_variants(settled, {}, search_dirs);
    const std::vector<Override> overrides = parse_overrides(settled);
    if (overrides.empty()) {
        return "this graph declares no cli overrides\n";
    }
    std::string out = "cli overrides:\n";
    for (const Override& o : overrides) {
        out +=
            fmt::format("  {:<28} -> {}{}\n", usage(o), target(o), o.required ? " (required)" : "");
    }
    return out;
}

} // namespace merian
