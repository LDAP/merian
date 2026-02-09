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

    // User disabled
    bool disable{};
    // Device does not support this node
    bool unsupported{};
    std::string unsupported_reason{};
    // Errors during build / connect
    std::vector<std::string> errors{};
    // Errors in on_connected and while run.
    std::vector<std::string> errors_queued{};

    // Cache input connectors (node->describe_inputs())
    // (on start_nodes added and checked for name conflicts)
    std::vector<InputConnectorHandle> input_connectors;
    std::unordered_map<std::string, InputConnectorHandle> input_connector_for_name;
    // Cache output connectors (node->describe_outputs())
    // (on conncet_nodes added and checked for name conflicts)
    std::vector<OutputConnectorHandle> output_connectors;
    std::unordered_map<std::string, OutputConnectorHandle> output_connector_for_name;

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

        uint32_t descriptor_set_binding{
            DescriptorSet::NO_DESCRIPTOR_BINDING}; // (on prepare_descriptor_sets)
        // precomputed such that (iteration % precomputed_resources.size()) is the index of the
        // resource that must be used in the iteration. Matches the descriptor_sets array below.
        // (resource handle, resource index the resources array of the corresponding output)
        // (on prepare_descriptor_sets)
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

        // precomputed occurrences in descriptor sets (needed to "record" descriptor set updates)
        // in descriptor sets of the node this output / resource belongs to
        std::vector<uint32_t> set_indices{};
        // in descriptor sets of other nodes this resource is accessed using inputs
        // (using in node, input connector, set_idx)
        std::vector<std::tuple<NodeHandle, InputConnectorHandle, uint32_t>> other_set_indices{};
    };
    struct PerOutputInfo {
        // (max_delay + 1) resources
        std::vector<PerResourceInfo> resources;
        std::vector<std::tuple<NodeHandle, InputConnectorHandle>> inputs;
        uint32_t descriptor_set_binding{
            DescriptorSet::NO_DESCRIPTOR_BINDING}; // (on prepare_descriptor_sets)
        // precomputed such that (iteration % precomputed_resources.size()) is the index of the
        // resource that must be used in the iteration. Matches the descriptor_sets array below.
        // (resource handle, resource index the resources array)
        std::vector<std::tuple<GraphResourceHandle, uint32_t>>
            precomputed_resources{}; // (on prepare_descriptor_sets)
    };
    std::unordered_map<OutputConnectorHandle, PerOutputInfo> output_connections{};

    // Precomputed descriptor set layout including all input and output connectors which
    // get_descriptor_info() does not return std::nullopt.
    DescriptorSetLayoutHandle descriptor_set_layout;

    // A descriptor set for each combination of resources that can occur, due to delayed accesses.
    // Also keep at least RING_SIZE to allow updating descriptor sets while iterations are in
    // flight. Access with iteration % data.descriptor_sets.size() (on prepare descriptor sets)
    std::vector<DescriptorSetHandle> descriptor_sets;
    std::vector<NodeIO> resource_maps;

    struct NodeStatistics {
        uint32_t last_descriptor_set_updates{};
    };
    NodeStatistics statistics{};

    void reset() {
        input_connectors.clear();
        output_connectors.clear();

        input_connector_for_name.clear();
        output_connector_for_name.clear();

        input_connections.clear();
        output_connections.clear();

        resource_maps.clear();
        descriptor_sets.clear();
        descriptor_set_layout.reset();

        statistics = {};

        errors.clear();
    }

    uint32_t set_index(const uint64_t run_iteration) const {
        assert(!descriptor_sets.empty());
        return run_iteration % descriptor_sets.size();
    }
};

inline std::string format_as(const NodeData::NodeStatistics stats) {
    return fmt::format("Descriptor bindings updated: {}", stats.last_descriptor_set_updates);
}

} // namespace graph_internal
} // namespace merian
