#pragma once

#include "merian/utils/hash.hpp"
#include "merian/utils/properties.hpp"

#include <unordered_set>

namespace merian_nodes {

// Describes the structure (nodes, connections) of a Graph and the configuration of the nodes. The
// GraphBuilder can take this description and build the runable graph from it.
class GraphPrototype {
  public:
    // Empty prototype
    GraphPrototype() {}

    // -----------------------------------------------------------------
    // -----------------------------------------------------------------

    /* Adds a node to the graph.
     *
     * The node_type must be a known type to the registry that is used to build the final graph. It
     * is not checked here!
     *
     * Throws invalid_argument, if a node with this identifier already exists.
     *
     * Returns the node identifier.
     */
    const std::string& add_node(const std::string& node_type,
                                const std::optional<std::string>& identifier = std::nullopt,
                                const nlohmann::json& config = {}) {}

    bool remove_node(const std::string& identifier) {}

    void add_connection(const std::string& src,
                        const std::string& dst,
                        const std::string& src_output,
                        const std::string& dst_input) {}

    bool remove_connection(const std::string& src,
                           const std::string& dst,
                           const std::string& dst_input) {}

    void set_node_config(const std::string& identifier, const nlohmann::json& config) {
        assert(nodes.contains(identifier));
        nodes.at(identifier).config = config;
    }

    const nlohmann::json& get_node_config(const std::string& identifier) const {
        assert(nodes.contains(identifier));
        return nodes.at(identifier).config;
    }

    // -----------------------------------------------------------------

    // Creates a graph structure from merian::Properties
    static void from_properties(merian::Properties& properties) {}

    // Dumps the graph structure to merian::Properties
    void to_properties(merian::Properties& properties) {}

  private:
    struct PerNodeInfo {
        const std::string node_type;

        nlohmann::json config;

        // (output_connector_name -> dst_node -> dst_input)
        std::map<std::string, std::map<std::string, std::string>> outgoing_connections;

        // (input connector name -> (src_node, src_output_name))
        std::unordered_map<std::string, std::pair<std::string, std::string>> incoming_connections;
    };

    // (identifier -> per_node_info)
    std::map<std::string, PerNodeInfo> nodes;

    // updated every time the structure changes (nodes and connections).
    // Node properties do not change the prototype, if they do not need a graph rebuild.
    uint64_t hash;
};

} // namespace merian_nodes
