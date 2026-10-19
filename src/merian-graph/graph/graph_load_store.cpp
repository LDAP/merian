#include "merian-graph/graph/graph.hpp"
#include "merian-graph/graph/graph_description.hpp"
#include "merian/utils/properties_json_load.hpp"

#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace merian {

std::vector<std::pair<std::string, std::string>> Graph::get_metadata() {
    // tEXt and COM are latin-1, so keep the config ascii-escaped
    return {{"Software", "merian " MERIAN_VERSION},
            {"merian.graph", to_description().to_json().dump(-1, ' ', true)}};
}

std::vector<std::string> Graph::from_description(const GraphDescription& description) {
    clear();
    settings = {};
    async_tasks.reset();

    loaded_description = description;

    std::vector<std::string> problems;
    const auto load_settings = [&](const nlohmann::json& properties, const std::string_view name,
                                   const auto& settings) {
        if (properties.empty()) {
            return;
        }
        if (const std::vector<std::string> load_problems =
                JSONLoadProperties::load_revealed(properties,
                                                  [&](Properties& props) {
                                                      settings(props);
                                                      return true;
                                                  });
            !load_problems.empty()) {
            problems.push_back(fmt::format("{}: {}", name, fmt::join(load_problems, "; ")));
            SPDLOG_WARN("{}", problems.back());
        }
    };
    load_settings(description.get_graph_properties(), "graph properties",
                  [&](Properties& props) { graph_settings(props); });
    load_settings(description.get_profiler_properties(), "profiler properties",
                  [&](Properties& props) { profiler_settings(props); });

    // Load nodes
    const auto& nodes = description.get_nodes();
    {
        // a node that removes itself leaves once everything is loaded
        const BusyScope busy_scope(*this);
        for (const auto& [identifier, node_info] : nodes) {
            const std::string& node_type = node_info.node_type;
            const nlohmann::json& config = node_info.config;
            const bool enabled = node_info.enabled;

            try {
                const NodeHandle node = registry.create_node_from_name(node_type);

                add_node(node, identifier);

                if (!config.empty()) {
                    if (auto problem = load_node_properties(identifier, node, config)) {
                        SPDLOG_WARN("{}", *problem);
                        problems.push_back(std::move(*problem));
                    }
                }

                node_data.at(node).enabled = enabled;
                node_data.at(node).linearization_order = node_info.linearization_order;

                SPDLOG_DEBUG("Loaded node '{}' of type '{}'", identifier, node_type);
            } catch (const std::exception& e) {
                SPDLOG_ERROR("Failed to load node '{}' of type '{}': {}", identifier, node_type,
                             e.what());
                throw;
            }
        }

        // Load connections
        for (const auto& [dst, node_info] : nodes) {
            for (const auto& [dst_input, source] : node_info.incoming_connections) {
                const auto& [src, src_output] = source;
                try {
                    add_connection(src, dst, src_output, dst_input);
                    SPDLOG_DEBUG("Loaded connection {} ({}) -> {} ({})", src, src_output, dst,
                                 dst_input);
                } catch (const std::exception& e) {
                    SPDLOG_ERROR("Failed to add connection {} ({}) -> {} ({}): {}", src, src_output,
                                 dst, dst_input, e.what());
                    throw;
                }
            }
        }
    }
    apply_deferred_edits();
    request_reconnect();

    SPDLOG_INFO("Graph loaded successfully with {} nodes", nodes.size());
    return problems;
}

GraphDescription Graph::to_description() {
    GraphDescription description;

    nlohmann::json properties = get_properties();
    description.set_graph_properties(properties["graph_properties"]);
    description.set_profiler_properties(properties["profiler"]);

    // Store top-level nodes: a parent's properties hold its children
    for (const NodeHandle& node : top_level_nodes()) {
        const std::string& identifier = node_data.at(node).identifier;
        const std::string& node_type = node_data.at(node).type_name;
        const nlohmann::json config = node->dump_config();
        const bool enabled = node_data.at(node).enabled;

        description.add_node(node_type, identifier, config);
        description.set_node_enabled(identifier, enabled);
        description.set_node_linearization_order(identifier,
                                                 node_data.at(node).linearization_order);
        // metadata only lives in the loaded description, not in live node state
        if (loaded_description.has_node(identifier)) {
            description.set_node_metadata(identifier,
                                          loaded_description.get_node_metadata(identifier));
        }

        SPDLOG_DEBUG("Storing node '{}' of type '{}'", identifier, node_type);
    }

    // Store connections
    for (const NodeHandle& node : top_level_nodes()) {
        const NodeData& data = node_data.at(node);
        for (const auto& [input, source] : data.desired_incoming_connections) {
            description.add_connection(node_data.at(source.node).identifier, data.identifier,
                                       source.name, input);
        }
    }

    description.set_cli(loaded_description.get_cli());

    SPDLOG_DEBUG("described graph with {} nodes", node_for_identifier.size());

    return description;
}

} // namespace merian
