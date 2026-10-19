#pragma once

#include "merian-graph/graph/errors.hpp"
#include "merian/io/file_loader.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace merian {

// Intermediate representation of a Graph which describes its structure (nodes, connections) and the
// configuration of the nodes. This representation is used to load and store graphs.
class GraphDescription {
  public:
    // -----------------------------------------------------------------
    // Data Structures
    // -----------------------------------------------------------------

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

        // (input connector name -> (src_node, src_output_name))
        std::map<std::string, std::pair<std::string, std::string>> incoming_connections{};
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

    // Throws invalid_argument if a node does not exist; the connectors are not checked.
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

    void set_node_enabled(const std::string& identifier, bool enabled) {
        assert(nodes.contains(identifier));
        nodes.at(identifier).enabled = enabled;
    }

    void set_node_linearization_order(const std::string& identifier, uint32_t order) {
        assert(nodes.contains(identifier));
        nodes.at(identifier).linearization_order = order;
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

    void set_cli(const nlohmann::json& value) {
        cli = value;
    }

    const nlohmann::json& get_cli() const {
        return cli;
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

    // Without a "version", the current one.
    static GraphDescription from_json(const nlohmann::json& json);

    void to_file(const std::filesystem::path& path) const {
        std::ofstream file(path.string());
        file << std::setw(4) << to_json() << '\n';
    }

    nlohmann::json to_json() const;

  public:
    // -----------------------------------------------------------------
    // Config files
    // -----------------------------------------------------------------

    // res/ and examples/ in the parents of the working directory, then the installed examples.
    static std::vector<std::filesystem::path> default_search_dirs();

    // The config at path, or else found in search_dirs. Throws config_error.
    static std::filesystem::path find_config(const std::filesystem::path& path,
                                             const std::vector<std::filesystem::path>& search_dirs);

    // Deep merge; "$+$key": [..] appends its (deduplicated) elements to the array at key.
    static void merge_into(nlohmann::json& base, const nlohmann::json& overwrite);

    // Throws std::invalid_argument if the identifier is empty, reserved (user, graph) or contains
    // reserved characters (., ->, /, :).
    static void validate_identifier(const std::string& identifier, const std::string& context);

    // "<base> <i>" with the first i that is not taken.
    static std::string unique_identifier(const std::string& base,
                                         const std::function<bool(const std::string&)>& taken);

  private:
    static void parse_graph_v4(const nlohmann::json& json, GraphDescription& description);

    void dump_graph_v4(nlohmann::json& json) const;

    // Connection parsing helpers
    static std::tuple<std::string, std::string, std::string>
    parse_arrow_connection(const std::string& connection, const std::string& node_id);

    static std::pair<std::string, std::string> parse_dot_target(const std::string& target,
                                                                const std::string& node_id,
                                                                const std::string& output_name);

  private:
    static constexpr const char* SCHEMA_VERSION_KEY = "version";
    static constexpr int SCHEMA_VERSION = 4;

    // (identifier -> per_node_info)
    std::map<std::string, PerNodeInfo> nodes;

    // Graph-level properties (iterations in flight, fps limiter, etc.)
    nlohmann::json graph_properties{};

    // Profiler properties
    nlohmann::json profiler_properties{};

    nlohmann::json cli{};
};

} // namespace merian
