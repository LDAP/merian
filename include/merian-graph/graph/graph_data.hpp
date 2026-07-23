#pragma once

#include "node.hpp"
#include "resource.hpp"

namespace merian {
namespace graph_internal {

// Describes a connection between two connectors of two nodes.
struct OutgoingNodeConnection {
    const NodeHandle dst;
    const std::string src_output;
    const std::string dst_input;

    bool operator==(const OutgoingNodeConnection&) const = default;

  public:
    struct Hash {
        size_t operator()(const OutgoingNodeConnection& c) const noexcept {
            return hash_val(c.dst, c.src_output, c.dst_input);
        }
    };
};

// Data that is stored for every node that is present in the graph.
struct NodeData {
    NodeData(const std::string& identifier) : identifier(identifier) {}

    // A unique name that identifies this node (user configurable).
    // This is not the name from the node registry.
    // (on add_node)
    std::string identifier;

    // User enabled
    bool enabled{true};
    // Device does not support this node
    bool unsupported{};
    std::string unsupported_reason{};
    // Errors during build / connect
    std::vector<std::string> errors{};
    // Errors in on_connected and while run.
    std::vector<std::string> errors_queued{};

    // (on cache_node_input_connectors / cache_node_output_connectors)
    std::unordered_map<ConnectorHandle, ConnectorAccess> connector_access;

    // Wiring declared by the input descriptors. (on cache_node_input_connectors)
    std::unordered_map<InputConnectorHandle, uint32_t> input_delay;
    std::unordered_map<InputConnectorHandle, bool> input_optional;

    // Shader cursor field name NodeIO::bind writes each connector to: in_<name> / out_<name>.
    // (on cache_node_input_connectors / cache_node_output_connectors)
    std::unordered_map<ConnectorHandle, std::string> bind_field_name;

    // Dependency layer: 0 for sources, else 1 + max over producers of non-delayed inputs.
    // (on build_layers)
    uint32_t level{0};

    // Cache input connectors (node->describe_inputs())
    // (on start_nodes added and checked for name conflicts)
    std::vector<InputConnectorHandle> input_connectors;
    std::unordered_map<std::string, InputConnectorHandle> input_connector_for_name;
    std::unordered_map<InputConnectorHandle, std::string> input_name_for_connector;
    // Cache output connectors (node->describe_outputs())
    // (on conncet_nodes added and checked for name conflicts)
    std::vector<OutputConnectorHandle> output_connectors;
    std::unordered_map<std::string, OutputConnectorHandle> output_connector_for_name;
    std::unordered_map<OutputConnectorHandle, std::string> output_name_for_connector;

    // --- Desired connections. ---
    // Set by the user using the public add_connection method.
    // This information is used by connect() to connect the graph
    std::unordered_set<OutgoingNodeConnection, typename OutgoingNodeConnection::Hash>
        desired_outgoing_connections;
    // (input connector name -> (src_node, src_output_name))
    std::unordered_map<std::string, std::pair<NodeHandle, std::string>>
        desired_incoming_connections;

    // --- Actual connections. ---
    // For each input the connected node and the corresponding output connector on the other
    // node (on connect).
    // For optional inputs an connection with nullptrs is inserted in start_nodes.
    struct PerInputInfo {
        NodeHandle node{};
        OutputConnectorHandle output{};

        // precomputed such that (iteration % precomputed_resources.size()) is the index of
        // the resource that must be used in the iteration. (resource handle, resource index in
        // the resources array of the corresponding output) (on precompute_resources)
        //
        // resources can be null if an optional input is not connected, the resource index is then
        // -1ul;
        std::vector<std::tuple<GraphResourceHandle, uint32_t>> precomputed_resources{};
    };
    std::unordered_map<InputConnectorHandle, PerInputInfo> input_connections{};
    // for each output the connected nodes and the corresponding input connector on the other
    // node (on connect)
    struct PerResourceInfo {
        GraphResourceHandle resource;
    };
    struct PerOutputInfo {
        // (max_delay + 1) resources
        std::vector<PerResourceInfo> resources;
        std::vector<std::tuple<NodeHandle, InputConnectorHandle>> inputs;
        // precomputed such that (iteration % precomputed_resources.size()) is the index of
        // the resource that must be used in the iteration. (on precompute_resources)
        std::vector<std::tuple<GraphResourceHandle, uint32_t>> precomputed_resources{};
    };
    std::unordered_map<OutputConnectorHandle, PerOutputInfo> output_connections{};

    std::vector<NodeIO> resource_maps;

    void reset() {
        connector_access.clear();
        input_delay.clear();
        input_optional.clear();
        bind_field_name.clear();
        input_connectors.clear();
        output_connectors.clear();

        input_connector_for_name.clear();
        output_connector_for_name.clear();
        input_name_for_connector.clear();
        output_name_for_connector.clear();

        input_connections.clear();
        output_connections.clear();

        resource_maps.clear();

        errors.clear();
    }

    uint32_t set_index(const uint64_t run_iteration) const {
        assert(!resource_maps.empty());
        return run_iteration % resource_maps.size();
    }
};

} // namespace graph_internal
} // namespace merian
