#pragma once

#include "merian/utils/properties.hpp"
#include "merian/utils/properties_json_dump.hpp"
#include "merian/utils/properties_json_load.hpp"

namespace merian_nodes {

// Describes the structure (nodes, connections) of a Graph and the configuration of the nodes. The
// GraphBuilder can take this description and build the runable graph from it.
class GraphDescription {
  public:
    // Empty graph
    GraphDescription() {}

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

    // Creates a graph description from merian::Properties
    static GraphDescription from_properties(merian::Properties& properties) {}

    static GraphDescription from_file(const std::filesystem::path& path) {
        merian::JSONLoadProperties props(path);
        return from_properties(props);
    }

    static GraphDescription from_file(const std::string& s_path) {
        std::filesystem::path path = s_path;
        return from_file(path);
    }

    // Dumps the graph description to merian::Properties
    void to_properties(merian::Properties& properties) {}

    void to_file(const std::filesystem::path& path) {
        merian::JSONDumpProperties props(path);
        to_properties(props);
    }

    void to_file(const std::string& s_path) {
        std::filesystem::path path = s_path;
        to_file(path);
    }

  private:
    struct PerNodeInfo {
        const std::string node_type;

        // ---------------------------------------------
        
        bool disabled = false;

        // Can be used to enforce a certain linearization of the graph.
        // Note, the driver might still move things around as the order is not enforced via barriers
        // by default.
        uint32_t linearization_order = 0;

        // ---------------------------------------------

        nlohmann::json config{};

        // (output_connector_name -> dst_node -> dst_input)
        std::map<std::string, std::map<std::string, std::string>> outgoing_connections{};

        // (input connector name -> src_node -> src_output_name)
        std::unordered_map<std::string, std::map<std::string, std::string>> incoming_connections{};
    };

    // (identifier -> per_node_info)
    std::map<std::string, PerNodeInfo> nodes;

    // updated every time the structure changes (nodes and connections).
    // Node properties do not change the prototype, if they do not need a graph rebuild.
    uint64_t hash;
};

} // namespace merian_nodes
