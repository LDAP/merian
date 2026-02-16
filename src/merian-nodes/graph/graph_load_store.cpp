#include "merian-nodes/graph/graph.hpp"
#include "merian-nodes/graph/graph_description.hpp"
#include "merian/utils/properties_json_dump.hpp"
#include "merian/utils/properties_json_load.hpp"

#include <spdlog/spdlog.h>

namespace merian {

// -----------------------------------------------------------------
// Public Load/Store Methods
// -----------------------------------------------------------------

void Graph::load_from_file(const std::filesystem::path& path) {
    SPDLOG_DEBUG("Loading graph from {}", path.string());
    merian_nodes::GraphDescription description = merian_nodes::GraphDescription::from_file(path);
    from_description(description);
}

void Graph::load_from_json(const nlohmann::json& json) {
    merian_nodes::GraphDescription description = merian_nodes::GraphDescription::from_json(json);
    from_description(description);
}

void Graph::store_to_file(const std::filesystem::path& path) {
    SPDLOG_DEBUG("Storing graph to {}", path.string());
    merian_nodes::GraphDescription description = to_description();
    description.to_file(path);
}

nlohmann::json Graph::store_to_json() {
    merian_nodes::GraphDescription description = to_description();
    return description.to_json();
}

// -----------------------------------------------------------------
// Private Helper Methods
// -----------------------------------------------------------------

void Graph::from_description(const merian_nodes::GraphDescription& description) {
    // Clear existing graph
    reset();

    // Load graph-level properties using JSONLoadProperties
    const auto& graph_props = description.get_graph_properties();
    if (!graph_props.empty()) {
        JSONLoadProperties load_props(graph_props);
        graph_properties(load_props);
    }

    // Load profiler properties using JSONLoadProperties
    const auto& profiler_props = description.get_profiler_properties();
    if (!profiler_props.empty()) {
        JSONLoadProperties load_props(profiler_props);
        profiler_properties(load_props);
    }

    // Load nodes
    const auto& nodes = description.get_nodes();
    for (const auto& [identifier, node_info] : nodes) {
        const std::string& node_type = node_info.node_type;
        const nlohmann::json& config = node_info.config;
        bool enabled = node_info.enabled;

        try {
            // Create node from registry
            NodeHandle node = NodeRegistry::get_instance().create_node_from_name(node_type);

            // Add node to graph (this initializes the node)
            add_node(node, identifier);

            // Load node configuration (after initialization)
            if (!config.empty()) {
                node->load_config(config);
            }

            // Set enabled state
            node_data.at(node).enabled = enabled;

            SPDLOG_DEBUG("Loaded node '{}' of type '{}'", identifier, node_type);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Failed to load node '{}' of type '{}': {}", identifier, node_type,
                         e.what());
            throw;
        }
    }

    // Load connections
    for (const auto& [src, node_info] : nodes) {
        for (const auto& [src_output, output_info] : node_info.outgoing_connections) {
            for (const auto& [dst, dst_inputs] : output_info.target) {
                for (const auto& dst_input : dst_inputs) {
                    try {
                        add_connection(src, dst, src_output, dst_input);
                        SPDLOG_DEBUG("Loaded connection {} ({}) -> {} ({})", src, src_output, dst,
                                     dst_input);
                    } catch (const std::exception& e) {
                        SPDLOG_ERROR("Failed to add connection {} ({}) -> {} ({}): {}", src,
                                     src_output, dst, dst_input, e.what());
                        throw;
                    }
                }
            }
        }
    }

    // Mark that we need to reconnect the graph
    needs_reconnect = true;

    SPDLOG_INFO("Graph loaded successfully with {} nodes", nodes.size());
}

merian_nodes::GraphDescription Graph::to_description() {
    merian_nodes::GraphDescription description;

    // Store graph-level properties using JSONDumpProperties
    {
        JSONDumpProperties dump_props;
        graph_properties(dump_props);
        description.set_graph_properties(dump_props.get());
    }

    // Store profiler properties using JSONDumpProperties
    {
        JSONDumpProperties dump_props;
        profiler_properties(dump_props);
        description.set_profiler_properties(dump_props.get());
    }

    // Store nodes
    for (const auto& [identifier, node] : node_for_identifier) {
        const std::string node_type = NodeRegistry::get_instance().node_type_name(node);
        const nlohmann::json config = node->dump_config();
        const bool enabled = node_data.at(node).enabled;

        description.add_node(node_type, identifier, config);
        description.set_node_enabled(identifier, enabled);

        SPDLOG_DEBUG("Storing node '{}' of type '{}'", identifier, node_type);
    }

    // Store connections
    for (const auto& [src, node] : node_for_identifier) {
        const auto& data = node_data.at(node);
        for (const auto& connection : data.desired_outgoing_connections) {
            const std::string& dst = node_data.at(connection.dst).identifier;
            description.add_connection(src, dst, connection.src_output, connection.dst_input);
            SPDLOG_DEBUG("Storing connection {} ({}) -> {} ({})", src, connection.src_output, dst,
                         connection.dst_input);
        }
    }

    SPDLOG_INFO("Graph stored successfully with {} nodes", node_for_identifier.size());

    return description;
}

} // namespace merian
