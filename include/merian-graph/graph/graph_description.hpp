#pragma once

#include "merian/io/file_loader.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>

namespace merian {

// Intermediate representation of a Graph which describes its structure (nodes, connections) and the
// configuration of the nodes. This representation is used to load and store graphs.
class GraphDescription {
  public:
    // -----------------------------------------------------------------
    // Data Structures
    // -----------------------------------------------------------------

    struct PerOutputInfo {
        // (dst_node -> set<dst_input>)
        std::map<std::string, std::set<std::string>> target;
        bool is_graph_output = false;
    };

    struct PerNodeInfo {
        std::string node_type;

        bool enabled = true;

        // Can be used to enforce a certain linearization of the graph.
        // Note, the driver might still move things around as the order is not enforced via barriers
        // by default.
        uint32_t linearization_order = 0;

        nlohmann::json config{};

        // Optional metadata for GUI editors (positions, colors, descriptions, etc.)
        nlohmann::json metadata{};

        // (output_connector_name -> output_info)
        std::map<std::string, PerOutputInfo> outgoing_connections{};

        // (input connector name -> (src_node, src_output_name))
        std::unordered_map<std::string, std::pair<std::string, std::string>> incoming_connections{};
    };

    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    // Empty graph
    GraphDescription() = default;

    // -----------------------------------------------------------------
    // Node and Connection Management
    // -----------------------------------------------------------------

    /* Adds a node to the graph.
     *
     * The node_type must be a known type to the registry that is used to build the final graph. It
     * is not checked here!
     *
     * Throws invalid_argument if a node with this identifier already exists.
     *
     * Returns the node identifier.
     */
    const std::string& add_node(const std::string& node_type,
                                const std::optional<std::string>& identifier = std::nullopt,
                                const nlohmann::json& config = {});

    /* Removes a node from the graph.
     *
     * Also removes all connections to/from this node.
     * Returns true if the node was removed, false if it didn't exist.
     */
    bool remove_node(const std::string& identifier);

    /* Adds a connection between two nodes.
     *
     * Does not check if the nodes exist or if the connection is valid.
     */
    void add_connection(const std::string& src,
                        const std::string& dst,
                        const std::string& src_output,
                        const std::string& dst_input);

    /* Removes a connection between two nodes.
     *
     * Returns true if the connection was removed, false if it didn't exist.
     */
    bool
    remove_connection(const std::string& src, const std::string& dst, const std::string& dst_input);

    // -----------------------------------------------------------------
    // Node Properties
    // -----------------------------------------------------------------

    void set_node_config(const std::string& identifier, const nlohmann::json& config) {
        assert(nodes.contains(identifier));
        nodes.at(identifier).config = config;
    }

    const nlohmann::json& get_node_config(const std::string& identifier) const {
        assert(nodes.contains(identifier));
        return nodes.at(identifier).config;
    }

    void set_node_enabled(const std::string& identifier, bool enabled) {
        assert(nodes.contains(identifier));
        nodes.at(identifier).enabled = enabled;
    }

    bool get_node_enabled(const std::string& identifier) const {
        assert(nodes.contains(identifier));
        return nodes.at(identifier).enabled;
    }

    void set_node_linearization_order(const std::string& identifier, uint32_t order) {
        assert(nodes.contains(identifier));
        nodes.at(identifier).linearization_order = order;
    }

    uint32_t get_node_linearization_order(const std::string& identifier) const {
        assert(nodes.contains(identifier));
        return nodes.at(identifier).linearization_order;
    }

    const std::string& get_node_type(const std::string& identifier) const {
        assert(nodes.contains(identifier));
        return nodes.at(identifier).node_type;
    }

    void set_node_metadata(const std::string& identifier, const nlohmann::json& metadata) {
        assert(nodes.contains(identifier));
        nodes.at(identifier).metadata = metadata;
    }

    const nlohmann::json& get_node_metadata(const std::string& identifier) const {
        assert(nodes.contains(identifier));
        return nodes.at(identifier).metadata;
    }

    // -----------------------------------------------------------------
    // Graph Properties
    // -----------------------------------------------------------------

    void set_graph_properties(const nlohmann::json& props) {
        graph_properties = props;
    }

    const nlohmann::json& get_graph_properties() const {
        return graph_properties;
    }

    void set_profiler_properties(const nlohmann::json& props) {
        profiler_properties = props;
    }

    const nlohmann::json& get_profiler_properties() const {
        return profiler_properties;
    }

    // -----------------------------------------------------------------
    // Access to Nodes and Connections
    // -----------------------------------------------------------------

    const std::map<std::string, PerNodeInfo>& get_nodes() const {
        return nodes;
    }

    bool has_node(const std::string& identifier) const {
        return nodes.contains(identifier);
    }

    // -----------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------

    static GraphDescription from_file(const std::filesystem::path& path) {
        assert(merian::FileLoader::exists(path));
        std::ifstream i(path.string());
        nlohmann::json json;
        i >> json;

        return from_json(json);
    }

    static GraphDescription from_json(const nlohmann::json& json);

    void to_file(const std::filesystem::path& path) const {
        std::ofstream file(path.string());
        file << std::setw(4) << to_json() << '\n';
    }

    nlohmann::json to_json() const;

  public:
    // Validates that an identifier doesn't contain reserved characters (., ->, /)
    // Throws std::invalid_argument if validation fails
    static void validate_identifier(const std::string& identifier, const std::string& context);

  private:
    static void parse_graph_v1(const nlohmann::json& json, GraphDescription& description);
    static void parse_graph_v2(const nlohmann::json& json, GraphDescription& description);
    static void parse_graph_v3(const nlohmann::json& json, GraphDescription& description);

    void dump_graph_v2(nlohmann::json& json) const;
    void dump_graph_v3(nlohmann::json& json) const;

    std::string generate_unique_identifier(const std::string& node_type);

    // Connection parsing helpers
    static std::tuple<std::string, std::string, std::string>
    parse_arrow_connection(const std::string& connection, const std::string& node_id);

    static std::pair<std::string, std::string> parse_dot_target(const std::string& target,
                                                                const std::string& node_id,
                                                                const std::string& output_name);

  private:
    static constexpr const char* SCHEMA_VERSION_KEY = "version";

    // (identifier -> per_node_info)
    std::map<std::string, PerNodeInfo> nodes;

    // Graph-level properties (iterations in flight, fps limiter, etc.)
    nlohmann::json graph_properties{};

    // Profiler properties
    nlohmann::json profiler_properties{};

    // Updated every time the structure changes (nodes and connections).
    // Node properties do not change the hash if they do not need a graph rebuild.
    uint64_t hash = 0;
};

} // namespace merian
