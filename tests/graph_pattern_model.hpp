#pragma once

#include "graph_pattern_nodes.hpp"

#include <random>
#include <ranges>

struct InputSpec {
    std::string src;
    uint32_t delay;
    std::string output = {};
};

struct NodeSpec {
    uint32_t id;
    PatternShape shape;
    std::map<std::string, InputSpec> inputs;
    bool enabled = true;
    bool stopped = false;
};

using GraphSpec = std::map<std::string, NodeSpec>;

inline NodeSpec& add_node(GraphSpec& spec, const std::string& identifier) {
    NodeSpec& node = spec[identifier];
    node.id = static_cast<uint32_t>(spec.size());
    return node;
}

inline std::string node_name(const uint32_t index) {
    return fmt::format("n{:02}", index);
}

inline uint32_t random_uint(std::mt19937& rng, const uint32_t min, const uint32_t max) {
    return std::uniform_int_distribution<uint32_t>(min, max)(rng);
}

inline void reshape_randomly(std::mt19937& rng, PatternShape& shape) {
    shape.width = random_uint(rng, 1, PatternNode::MAX_EXTENT);
    shape.height = random_uint(rng, 1, PatternNode::MAX_EXTENT);
    shape.buffer_words = random_uint(rng, 1, PatternNode::SLOT_SIZE / sizeof(uint32_t));
    shape.scratch = random_uint(rng, 0, 1) == 0;
    shape.fail_outputs = random_uint(rng, 0, 14) == 0;
    shape.fail_connected = random_uint(rng, 0, 14) == 0;
}

inline bool is_version(const std::string& output) {
    return output == "inp" || output == "binp";
}

// The output a version continues.
inline std::string root_output(const std::string& output) {
    return !is_version(output) ? output : output == "inp" ? "out" : "buf";
}

inline std::string source_output(const std::string& input, const InputSpec& source) {
    if (!source.output.empty()) {
        return source.output;
    }
    return is_image_input(input) ? "out" : "buf";
}

inline bool
written_in_place(const GraphSpec& spec, const std::string& src, const std::string& output) {
    const std::string input = output == "out" || output == "inp" ? "inp" : "binp";
    return std::ranges::any_of(spec, [&](const auto& entry) {
        const auto source = entry.second.inputs.find(input);
        return source != entry.second.inputs.end() && source->second.src == src &&
               source_output(input, source->second) == output;
    });
}

inline void connect_randomly(std::mt19937& rng, GraphSpec& spec, const uint32_t index) {
    const std::string identifier = node_name(index);
    NodeSpec& node = spec.at(identifier);
    node.inputs.clear();
    const auto pick = [&](const std::vector<InputSpec>& sources) {
        return sources[random_uint(rng, 0, static_cast<uint32_t>(sources.size()) - 1)];
    };
    const auto sources_for = [&](const uint32_t delay, const std::string& version) {
        std::vector<InputSpec> sources;
        for (const auto& [src_identifier, src] : spec) {
            if (delay == 0 ? src.id <= index : !src.shape.persistent) {
                sources.push_back({src_identifier, delay});
                if (src.shape.in_place) {
                    sources.push_back({src_identifier, delay, version});
                }
            }
        }
        return sources;
    };
    for (uint32_t i = 0; i < INPUT_COUNT; i++) {
        const uint32_t delay = random_uint(rng, 0, 9) < 7 ? 0 : random_uint(rng, 1, 2);
        node.shape.delays[i] = delay;
        const std::vector<InputSpec> sources =
            sources_for(delay, is_image_input(INPUT_NAMES[i]) ? "inp" : "binp");
        const bool connected = random_uint(rng, 0, 9) >= 4 && !sources.empty();
        node.shape.required[i] =
            connected ? random_uint(rng, 0, 9) < 3 : random_uint(rng, 0, 19) == 0;
        if (connected) {
            node.inputs[INPUT_NAMES[i]] = pick(sources);
        }
    }
    if (node.shape.in_place) {
        for (const std::string version : {"inp", "binp"}) {
            const std::vector<InputSpec> sources = sources_for(0, version);
            std::vector<InputSpec> unwritten;
            std::ranges::copy_if(sources, std::back_inserter(unwritten), [&](const auto& source) {
                return !written_in_place(spec, source.src, source_output(version, source));
            });
            if (!sources.empty() && random_uint(rng, 0, 9) < 7) {
                node.inputs[version] =
                    pick(!unwritten.empty() && random_uint(rng, 0, 4) != 0 ? unwritten : sources);
            }
        }
    }
    node.shape.pass_through = node.shape.delays[0] == 0 &&
                              node.shape.delays[IMAGE_INPUT_COUNT] == 0 &&
                              random_uint(rng, 0, 4) != 0;
}

// Disabled nodes pass through, writers that do not run pass on the version they received.
inline std::pair<std::string, std::string> resolve(const GraphSpec& spec,
                                                   const std::set<std::string>& running,
                                                   std::string src,
                                                   std::string output) {
    for (std::set<std::pair<std::string, std::string>> passed;
         passed.emplace(src, output).second;) {
        const NodeSpec& node = spec.at(src);
        std::string passing;
        if (is_version(output) && !running.contains(src)) {
            passing = output;
        } else if (!is_version(output) && node.shape.pass_through && !node.enabled) {
            passing = output == "out" ? "in0" : "bin0";
        }
        const auto input = node.inputs.find(passing);
        if (input == node.inputs.end()) {
            break;
        }
        output = source_output(passing, input->second);
        src = input->second.src;
    }
    return {src, output};
}

inline std::optional<std::string> root_of(const GraphSpec& spec,
                                          const std::set<std::string>& running,
                                          std::string src,
                                          std::string output) {
    while (running.contains(src)) {
        if (!is_version(output)) {
            return src;
        }
        const auto source = spec.at(src).inputs.find(output);
        if (source == spec.at(src).inputs.end()) {
            return std::nullopt;
        }
        std::tie(src, output) =
            resolve(spec, running, source->second.src, source_output(output, source->second));
    }
    return std::nullopt;
}

struct Read {
    std::string src;
    std::string output;
    uint32_t delay;
};

inline std::map<std::string, Read>
connected_reads(const GraphSpec& spec, const std::set<std::string>& running, const NodeSpec& node) {
    std::map<std::string, Read> reads;
    for (const auto& [input, source] : node.inputs) {
        const auto [src, output] = resolve(spec, running, source.src, source_output(input, source));
        const std::optional<std::string> root = root_of(spec, running, src, output);
        const bool self_loop = source.delay == 0 && spec.at(src).id == node.id;
        if (root && !self_loop && (source.delay == 0 || !spec.at(*root).shape.persistent)) {
            reads.emplace(input, Read{src, output, source.delay});
        }
    }
    return reads;
}

using Reads = std::map<std::string, std::map<std::string, Read>>;

// Writers that would have to run after a reader of their version that depends on them, or that read
// it themselves.
inline std::set<std::string> cyclic_writers(const Reads& reads) {
    std::map<std::string, std::set<std::string>> successors;
    std::map<std::string, std::set<std::string>> readers;
    std::set<std::string> cyclic;
    for (const auto& [node, node_reads] : reads) {
        for (const auto& [input, read] : node_reads) {
            if (read.delay == 0) {
                successors[read.src].insert(node);
            }
        }
    }
    for (const auto& [writer, writer_reads] : reads) {
        for (const auto& [version_input, version] : writer_reads) {
            if (!is_version(version_input)) {
                continue;
            }
            for (const auto& [reader, reader_reads] : reads) {
                for (const auto& [input, read] : reader_reads) {
                    if (is_version(input) || read.delay != 0 || read.src != version.src ||
                        read.output != version.output) {
                        continue;
                    }
                    if (reader == writer) {
                        cyclic.insert(writer);
                    } else {
                        successors[reader].insert(writer);
                        readers[writer].insert(reader);
                    }
                }
            }
        }
    }
    for (const auto& [writer, writer_readers] : readers) {
        std::set<std::string> reached;
        for (std::vector<std::string> stack{writer}; !stack.empty();) {
            const std::string node = stack.back();
            stack.pop_back();
            for (const std::string& successor : successors[node]) {
                if (reached.insert(successor).second) {
                    stack.push_back(successor);
                }
            }
        }
        if (std::ranges::any_of(writer_readers, [&](const std::string& reader) {
                return reached.contains(reader);
            })) {
            cyclic.insert(writer);
        }
    }
    return cyclic;
}

struct Outcome {
    std::set<std::string> running;
    std::set<std::string> rejected;
};

// Drops nodes missing a required input or on a cycle of undelayed reads, then applies the in-place
// rules in the graph's order: cyclic writers, versions written twice, delayed reads.
inline void
settle(const GraphSpec& spec, std::set<std::string>& running, std::set<std::string>& rejected) {
    for (bool changed = true; changed;) {
        changed = false;
        for (auto it = running.begin(); it != running.end();) {
            const NodeSpec& node = spec.at(*it);
            const std::map<std::string, Read> reads = connected_reads(spec, running, node);
            bool runs = !(node.shape.in_place && node.shape.required_in_place) ||
                        (reads.contains("inp") && reads.contains("binp"));
            for (uint32_t i = 0; i < INPUT_COUNT; i++) {
                runs &= !node.shape.required[i] || reads.contains(INPUT_NAMES[i]);
            }
            if (runs) {
                it++;
            } else {
                it = running.erase(it);
                changed = true;
            }
        }
        if (changed) {
            continue;
        }

        // nodes on a cycle of undelayed reads
        std::map<std::string, std::set<std::string>> reads_from;
        for (const std::string& node : running) {
            for (const auto& [input, read] : connected_reads(spec, running, spec.at(node))) {
                if (read.delay == 0) {
                    reads_from[node].insert(read.src);
                }
            }
        }
        std::set<std::string> stuck;
        for (const std::string& node : running) {
            std::set<std::string> reached;
            for (std::vector<std::string> stack{node}; !stack.empty();) {
                const std::string current = stack.back();
                stack.pop_back();
                for (const std::string& src : reads_from[current]) {
                    if (reached.insert(src).second) {
                        stack.push_back(src);
                    }
                }
            }
            if (reached.contains(node)) {
                stuck.insert(node);
            }
        }
        if (!stuck.empty()) {
            for (const std::string& node : stuck) {
                running.erase(node);
            }
            changed = true;
            continue;
        }

        Reads reads;
        std::map<std::pair<std::string, std::string>, std::vector<std::string>> writers;
        for (const std::string& identifier : running) {
            reads[identifier] = connected_reads(spec, running, spec.at(identifier));
            for (const auto& [input, read] : reads[identifier]) {
                if (is_version(input)) {
                    writers[{read.src, read.output}].push_back(identifier);
                }
            }
        }
        std::set<std::string> errors = cyclic_writers(reads);
        if (errors.empty()) {
            for (const auto& [version, version_writers] : writers) {
                if (version_writers.size() > 1) {
                    errors.insert(version_writers.begin(), version_writers.end());
                }
            }
        }
        if (errors.empty()) {
            for (const auto& [node, node_reads] : reads) {
                for (const auto& [input, read] : node_reads) {
                    if (read.delay > 0 && writers.contains({read.src, read.output})) {
                        errors.insert(node);
                    }
                }
            }
        }
        for (const std::string& identifier : errors) {
            running.erase(identifier);
        }
        rejected.insert(errors.begin(), errors.end());
        changed = !errors.empty();
    }
}

// Failed nodes stay failed until a connect raises no new error.
inline Outcome running_nodes(const GraphSpec& spec) {
    Outcome outcome;
    std::set<std::string> failed;
    for (;;) {
        std::set<std::string> running;
        for (const auto& [identifier, node] : spec) {
            if (node.enabled && !node.stopped && !node.shape.fail_outputs &&
                !failed.contains(identifier) && !outcome.rejected.contains(identifier)) {
                running.insert(identifier);
            }
        }
        settle(spec, running, outcome.rejected);
        const size_t failed_count = failed.size();
        std::ranges::copy_if(
            running, std::inserter(failed, failed.end()), [&](const std::string& node) {
                return spec.at(node).shape.fail_connected || spec.at(node).shape.fail_process;
            });
        if (failed.size() == failed_count) {
            outcome.running = std::move(running);
            return outcome;
        }
    }
}

inline GraphSpec random_spec(std::mt19937& rng, const uint32_t node_count) {
    GraphSpec spec;
    for (uint32_t index = 0; index < node_count; index++) {
        PatternShape& shape = add_node(spec, node_name(index)).shape;
        reshape_randomly(rng, shape);
        shape.persistent = random_uint(rng, 0, 4) == 0;
        shape.in_place = random_uint(rng, 0, 2) == 0;
        shape.required_in_place = shape.in_place && random_uint(rng, 0, 3) == 0;
    }
    for (uint32_t index = 0; index < node_count; index++) {
        connect_randomly(rng, spec, index);
    }
    return spec;
}

inline json shape_properties(const uint32_t id, const PatternShape& shape) {
    json properties = {{"id", id},
                       {"width", shape.width},
                       {"height", shape.height},
                       {"buffer words", shape.buffer_words},
                       {"persistent", shape.persistent},
                       {"scratch", shape.scratch},
                       {"pass through", shape.pass_through},
                       {"in place", shape.in_place},
                       {"required in place", shape.required_in_place},
                       {"fail outputs", shape.fail_outputs},
                       {"fail connected", shape.fail_connected},
                       {"fail process", shape.fail_process},
                       {"counting", shape.counting},
                       {"sampled", shape.sampled},
                       {"reconnect on change", shape.reconnect_on_change},
                       {"defaults", shape.defaults}};
    for (uint32_t i = 0; i < INPUT_COUNT; i++) {
        properties["delay " + INPUT_NAMES[i]] = shape.delays[i];
        properties["required " + INPUT_NAMES[i]] = shape.required[i];
    }
    return properties;
}

inline json describe(const GraphSpec& spec, const bool alias, const uint32_t iterations_in_flight) {
    json nodes = json::object();
    for (const auto& [identifier, node] : spec) {
        nodes[identifier] = {{"type", "Pattern"},
                             {"enabled", node.enabled},
                             {"properties", shape_properties(node.id, node.shape)}};
    }
    for (const auto& [identifier, node] : spec) {
        for (const auto& [input, src] : node.inputs) {
            const std::string output = source_output(input, src);
            nodes[src.src]["outputs"].push_back(
                fmt::format("{}->{}.{}", output, identifier, input));
        }
    }
    return {
        {"version", 4},
        {"graph_properties",
         {{"alias transient resources", alias}, {"iterations in flight", iterations_in_flight}}},
        {"nodes", nodes}};
}

struct GroupSpec {
    std::vector<NodeSpec> children;
    // at most one
    std::vector<GroupSpec> inner{};
    PatternGroup::Settings settings{};
};

struct GroupedSpec {
    GraphSpec nodes;
    std::map<std::string, GroupSpec> groups;
};

inline json group_properties(const GroupSpec& group) {
    const PatternGroup::Settings& settings = group.settings;
    json properties = {{"children", group.children.size()},
                       {"inner group", !group.inner.empty()},
                       {"prefix", settings.prefix},
                       {"broken", settings.broken},
                       {"parallel", settings.parallel},
                       {"public in1", settings.public_in1},
                       {"own input", settings.own_input},
                       {"own delay", settings.own_delay},
                       {"own required", settings.own_required},
                       {"own output", settings.own_output}};
    for (uint32_t i = 0; i < group.children.size(); i++) {
        properties[fmt::format("child {}", i)] =
            shape_properties(group.children[i].id, group.children[i].shape);
    }
    if (!group.inner.empty()) {
        properties["inner"] = group_properties(group.inner.front());
    }
    return properties;
}

inline json
describe(const GroupedSpec& grouped, const bool alias, const uint32_t iterations_in_flight) {
    json description = describe(grouped.nodes, alias, iterations_in_flight);
    for (const auto& [identifier, group] : grouped.groups) {
        description["nodes"][identifier]["type"] = "Pattern Group";
        description["nodes"][identifier]["properties"] = group_properties(group);
    }
    return description;
}

inline bool describes_subgraph(const GroupSpec& group) {
    return group.settings.broken == static_cast<uint32_t>(PatternGroup::Broken::NO);
}

inline bool group_runs(const GroupSpec& group, const NodeSpec& node) {
    const PatternGroup::Settings& settings = group.settings;
    return node.enabled && settings.broken == static_cast<uint32_t>(PatternGroup::Broken::NO) &&
           !settings.own_output &&
           (!settings.own_required || node.inputs.contains(settings.own_input));
}

using ConnectorRef = std::pair<std::string, std::string>;

// Returns the connector each public output carries.
inline std::map<std::string, ConnectorRef>
add_group(GraphSpec& flat,
          std::set<std::string>& nodes,
          const std::string& identifier,
          const GroupSpec& group,
          const std::map<std::string, ConnectorRef>& sources,
          const bool runs) {
    const auto source = [&](const std::string& connector) -> std::optional<ConnectorRef> {
        const auto it = sources.find(connector);
        return it == sources.end() ? std::nullopt : std::optional<ConnectorRef>(it->second);
    };
    const PatternGroup::Settings& settings = group.settings;
    std::map<std::string, ConnectorRef> exports;
    std::optional<ConnectorRef> out = source("in0");
    std::optional<ConnectorRef> buf = source("bin0");
    bool writer_found = false;
    for (uint32_t i = 0; i < group.children.size(); i++) {
        const std::string name = fmt::format("{}:{}{}", identifier, settings.prefix, i);
        NodeSpec node = group.children[i];
        node.stopped = !runs;
        node.inputs.clear();
        const auto connect = [&](const std::string& input,
                                 const std::optional<ConnectorRef>& from) {
            const auto index =
                static_cast<uint32_t>(std::ranges::find(INPUT_NAMES, input) - INPUT_NAMES.begin());
            if (from) {
                node.inputs[input] = {
                    from->first, index < INPUT_COUNT ? node.shape.delays[index] : 0, from->second};
            }
        };
        if (settings.parallel) {
            out = source("in0");
            buf = source("bin0");
        }
        connect("in0", out);
        connect("bin0", buf);
        connect("in2", source("in2"));
        // an unwired public input leaves the auto connection in place
        if (settings.public_in1 && i == 0 && (source("in2") || !node.shape.defaults)) {
            connect("in1", source("in2"));
        } else if (node.shape.defaults && node.shape.delays[1] > 0) {
            connect("in1", ConnectorRef{name, "out"});
        }
        if (node.shape.defaults && node.shape.delays[4] > 0) {
            connect("bin1", buf);
        }
        if (node.shape.in_place && !writer_found) {
            writer_found = true;
            for (const std::string connector : {"inp", "binp"}) {
                connect(connector, source(connector));
                exports[connector] = {name, connector};
            }
        }
        flat[name] = node;
        nodes.insert(name);
        out = ConnectorRef{name, "out"};
        buf = ConnectorRef{name, "buf"};
    }
    if (!group.inner.empty()) {
        if (settings.parallel) {
            out = source("in0");
            buf = source("bin0");
        }
        std::map<std::string, ConnectorRef> inner_sources;
        for (const auto& [connector, from] : {std::pair{"in0", out},
                                              {"bin0", buf},
                                              {"in2", source("in2")},
                                              {"inp", source("inp")},
                                              {"binp", source("binp")}}) {
            if (from && (!is_version(connector) || !writer_found)) {
                inner_sources[connector] = *from;
            }
        }
        const std::string inner_identifier =
            fmt::format("{}:{}{}", identifier, settings.prefix, group.children.size());
        nodes.insert(inner_identifier);
        const std::map<std::string, ConnectorRef> inner_exports =
            add_group(flat, nodes, inner_identifier, group.inner.front(), inner_sources, runs);
        out = inner_exports.at("out");
        buf = inner_exports.at("buf");
        for (const std::string connector : {"inp", "binp"}) {
            if (!writer_found && inner_exports.contains(connector)) {
                exports[connector] = inner_exports.at(connector);
            }
        }
    }
    exports["out"] = *out;
    exports["buf"] = *buf;
    return exports;
}

struct FlatSpec {
    GraphSpec spec;
    std::set<std::string> nodes;
};

// Disabled groups become pass-through Pattern nodes, broken ones nodes without connectors.
inline FlatSpec flatten(const GroupedSpec& grouped) {
    // what the public outputs carry does not depend on the sources
    std::map<std::string, std::map<std::string, ConnectorRef>> exports;
    for (const auto& [identifier, group] : grouped.groups) {
        GraphSpec scratch;
        std::set<std::string> scratch_nodes;
        exports[identifier] = add_group(scratch, scratch_nodes, identifier, group, {}, true);
        if (!describes_subgraph(group)) {
            for (const std::string connector : {"out", "buf", "inp", "binp"}) {
                exports[identifier][connector] = {identifier, connector};
            }
        } else if (!grouped.nodes.at(identifier).enabled) {
            exports[identifier]["out"] = {identifier, "out"};
            exports[identifier]["buf"] = {identifier, "buf"};
        }
    }
    const auto flat_source = [&](const std::string& input, const InputSpec& source) {
        const std::string output = source_output(input, source);
        if (grouped.groups.contains(source.src)) {
            return exports.at(source.src).at(output);
        }
        return ConnectorRef{source.src, output};
    };

    FlatSpec flat;
    for (const auto& [identifier, node] : grouped.nodes) {
        flat.nodes.insert(identifier);
        NodeSpec flat_node = node;
        for (auto& [input, source] : flat_node.inputs) {
            std::tie(source.src, source.output) = flat_source(input, source);
        }
        if (!grouped.groups.contains(identifier)) {
            flat.spec[identifier] = flat_node;
            continue;
        }
        const GroupSpec& group = grouped.groups.at(identifier);
        std::map<std::string, ConnectorRef> sources;
        for (const auto& [connector, source] : node.inputs) {
            sources[connector] = flat_source(connector, source);
        }
        add_group(flat.spec, flat.nodes, identifier, group, sources, group_runs(group, node));
        const bool described = describes_subgraph(group);
        if (!node.enabled || !described) {
            flat_node.enabled = false;
            flat_node.shape = {.pass_through = described};
            std::erase_if(flat_node.inputs, [&](const auto& entry) {
                return !described || (entry.first != "in0" && entry.first != "bin0");
            });
            flat.spec[identifier] = flat_node;
        }
    }
    return flat;
}

inline GroupedSpec subgraph_spec() {
    GroupedSpec grouped;
    GraphSpec& nodes = grouped.nodes;
    add_node(nodes, "a");
    add_node(nodes, "a2");
    add_node(nodes, "x").inputs = {{"in0", {"a2", 0}}};
    add_node(nodes, "g").inputs = {{"in0", {"a", 0}},
                                   {"bin0", {"a", 0}},
                                   {"in2", {"a", 0}},
                                   {"inp", {"a2", 0}},
                                   {"binp", {"a2", 0}}};
    add_node(nodes, "h").inputs = {{"in0", {"g", 0}}, {"bin0", {"g", 0}}};
    NodeSpec& reader = add_node(nodes, "r");
    reader.inputs = {{"in0", {"h", 0}},
                     {"bin0", {"h", 0}},
                     {"in1", {"g", 1}},
                     {"in2", {"g", 0, "inp"}},
                     {"bin1", {"g", 0, "binp"}}};
    reader.shape.delays[1] = 1;

    uint32_t id = 100;
    const auto child = [&](const PatternShape& shape) { return NodeSpec{id++, shape, {}}; };
    const PatternShape writer{.in_place = true};
    PatternShape with_defaults{.defaults = true};
    with_defaults.delays[1] = 1;
    with_defaults.delays[4] = 2;
    PatternShape delayed;
    delayed.delays[2] = 1;
    grouped.groups["g"] = {.children = {child(writer), child(with_defaults)},
                           .inner = {{.children = {child(with_defaults), child(delayed)}}}};
    grouped.groups["h"] = {.children = {child(with_defaults)}};
    return grouped;
}
