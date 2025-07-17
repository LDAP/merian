#pragma once

#include "merian/io/file_loader.hpp"
#include "merian/utils/properties_json_load.hpp"

#include <fstream>

namespace merian_nodes {

// Intermediate representation of a Graph which describes its structure (nodes, connections) and the
// configuration of the nodes. This representation is used to load an store graphs.
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

    // -----------------------------------------------------------------

    const nlohmann::json& get_node_config(const std::string& identifier) const {
        assert(nodes.contains(identifier));
        return nodes.at(identifier).config;
    }

    // -----------------------------------------------------------------

    static GraphDescription from_file(const std::filesystem::path& path) {
        assert(merian::FileLoader::exists(path));
        std::ifstream i(path.string());
        nlohmann::json json;
        i >> json;

        return from_json(json);
    }

    static GraphDescription from_json(const nlohmann::json& json) {
        GraphDescription description;
        if (!json.contains(SCHEMA_VERSION_KEY)) {
            parse_graph_v1(json, description);
        }

        int schema_version = json[SCHEMA_VERSION_KEY].get<int>();
        if (schema_version == 2) {
            parse_graph_v2(json, description);
        } else {
            throw std::runtime_error{fmt::format("schema version {} unsupported.", schema_version)};
        }

        return description;
    }

    // -----------------------------------------------------------------

    void to_file(const std::filesystem::path& path) {
        std::ofstream file(path.string());
        file << std::setw(4) << to_json() << '\n';
    }

    nlohmann::json to_json() {
        // merian::JSONDumpProperties props;
        // to_properties(props);

        // return props.get();
    }

  private:
    static void parse_graph_v1(const nlohmann::json& json, GraphDescription& description) {
        merian::JSONLoadProperties props(json);
        // ...
    }

    static void parse_graph_v2(const nlohmann::json& json, GraphDescription& description) {
        merian::JSONLoadProperties props(json);
        // ...
    }

  private:
    static constexpr int SCHEMA_VERSION = 2;
    static constexpr std::string SCHEMA_VERSION_KEY = "schema_version";

    struct PerOutputInfo {
        // (dst_node -> dst_input)
        std::map<std::string, std::string> target;
        bool is_graph_output = false;
    };

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

        // (output_connector_name -> output_info)
        std::map<std::string, PerOutputInfo> outgoing_connections{};

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
