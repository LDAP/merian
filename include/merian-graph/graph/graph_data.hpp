#pragma once

#include "node.hpp"
#include "resource.hpp"

#include "merian/vk/memory/memory_allocator_tracking.hpp"
#include "merian/vk/memory/memory_allocator_transient.hpp"

#include <algorithm>
#include <optional>
#include <tuple>

namespace merian {

// Whether a node runs: switched off or unsupported now, else what the last connect made of it.
enum class NodeState : uint8_t {
    OK,
    // switched off by the user
    DISABLED,
    // disabled, consumers receive what it passes through
    BYPASSED,
    // the device cannot run it
    UNSUPPORTED,
    // a node it depends on contributes nothing
    FORCE_DISABLED,
    ERRORED,
};

namespace graph_internal {

struct Port {
    NodeHandle node;
    std::string name;

    bool operator==(const Port&) const = default;
};

struct Connection {
    NodeHandle src;
    std::string src_output;
    NodeHandle dst;
    std::string dst_input;
};

struct InputDecision {
    enum class Kind : uint8_t {
        // for the producer to describe its outputs
        WAIT,
        CONNECTED,
        UNCONNECTED,
        // the producer contributes nothing
        STOPPED,
        // the wiring cannot work for any input
        REJECTED,
        // this input cannot read the source
        INCOMPATIBLE,
    };

    Kind kind;
    // WAIT, CONNECTED
    NodeHandle producer{};
    // CONNECTED
    OutputConnectorHandle output{};
    // STOPPED, REJECTED, INCOMPATIBLE
    std::string reason{};
    // REJECTED, INCOMPATIBLE: the desired connection to remove
    std::optional<Connection> removable{};
};

struct ResourceData {
    struct Copy {
        GraphResourceHandle resource;
        TransientMemoryScopeHandle scope;
        // A delayed input reads it before its producer first wrote it.
        bool read_before_written;

        bool aliased() const {
            return std::ranges::any_of(scope->get_placements(),
                                       &TransientMemoryAllocator::Placement::aliased);
        }
    };
    // One, the period, or max delay + 1 without aliasing: copy c is written in the iterations i
    // with i % copies.size() == c.
    std::vector<Copy> copies;
};

inline bool reads_before_writer(const InputConnectorDescriptor& descriptor) {
    return !descriptor.in_place && descriptor.delay == 0;
}

// Rebuilt on every connect.
struct NodeConnectState {
    std::vector<std::string> errors;
    // A required input's producer contributes nothing, or the parent does not run; the reason names
    // the chain's origin.
    bool force_disabled{};
    std::string force_disabled_reason;
    bool bypassed{};
    // Its outputs are described.
    bool visited{};
    // Dependency layer: 0 for sources, else 1 + max over its predecessors (on build_layers)
    uint32_t level{0};

    // subgraph of a parent without errors
    std::vector<Connection> subgraph_connections;
    // (public input -> the children's inputs it feeds)
    std::unordered_map<std::string, std::vector<Port>> public_inputs;
    // (public output -> the child's output it carries)
    std::unordered_map<std::string, Port> public_outputs;
    // (public output -> public input that passes through to it)
    std::unordered_map<std::string, std::string> public_pass_throughs;

    // subgraph of a child
    // (input -> the output of another child connected to it)
    std::unordered_map<std::string, Port> subgraph_sources;
    // (input -> the public input of the parent that feeds it)
    std::unordered_map<std::string, std::string> public_feeds;

    // Shader cursor field name of each connector: in_<name> / out_<name>.
    std::unordered_map<ConnectorHandle, std::string> bind_field_name;

    // In declaration order.
    std::vector<InputConnectorHandle> input_connectors;
    std::unordered_map<std::string, InputConnectorHandle> input_connector_for_name;
    std::unordered_map<InputConnectorHandle, InputConnectorDescriptor> input_descriptors;
    // (output name -> name of the input that passes through to it)
    std::unordered_map<std::string, std::string> pass_through_inputs;
    // In declaration order, then the versions of in-place inputs.
    std::vector<OutputConnectorHandle> output_connectors;
    std::unordered_map<std::string, OutputConnectorHandle> output_connector_for_name;
    std::unordered_map<OutputConnectorHandle, OutputConnectorDescriptor> output_descriptors;

    // For each input the connected node and output, null for an optional input that stays
    // unconnected.
    struct PerInputInfo {
        NodeHandle node{};
        OutputConnectorHandle output{};

        // (resource, copy index) per set, (iteration % period) is the set of an iteration; null and
        // -1ul if unconnected (on precompute_resources)
        std::vector<std::tuple<GraphResourceHandle, uint32_t>> precomputed_resources{};
    };
    std::unordered_map<InputConnectorHandle, PerInputInfo> input_connections{};
    struct PerOutputInfo {
        // The in-place input it continues, null for a declared output.
        InputConnectorHandle version_of;
        // null if the output is disabled; versions share their root's (on plan_memory)
        std::shared_ptr<ResourceData> resource_data;
        std::vector<std::tuple<NodeHandle, InputConnectorHandle>> inputs;
        // (resource, copy index) per set; null and -1ul if disabled (on precompute_resources)
        std::vector<std::tuple<GraphResourceHandle, uint32_t>> precomputed_resources{};
    };
    std::unordered_map<OutputConnectorHandle, PerOutputInfo> output_connections{};

    std::vector<NodeIO> resource_maps;
};

struct NodeData : NodeConnectState {
    NodeData(const std::string& identifier, const std::string& type_name) : type_name(type_name) {
        set_identifier(identifier);
    }

    // A unique name that identifies this node, "<parent>:<name>" for a child.
    std::string identifier;
    // The name in the node registry.
    std::string type_name;
    std::string label;

    TrackingMemoryAllocatorHandle memory_tracker;
    ResourceAllocatorHandle allocator;

    // User enabled
    bool enabled{true};
    // Within-layer execution order (lower runs earlier, ties by identifier)
    uint32_t linearization_order{0};
    bool unsupported{};
    std::string unsupported_reason{};
    // Errors in on_connected and while running, and rejections that restart connecting; cleared by
    // a connect that needs no reconnect.
    std::vector<std::string> errors_queued{};

    NodeHandle parent;
    std::vector<NodeHandle> children;

    // (input -> output) connected by the user or an auto connection
    std::unordered_map<std::string, Port> desired_incoming_connections;
    // Inputs whose auto connection was made, or found connected otherwise.
    std::unordered_set<std::string> settled_auto_connections;
    // Auto connected inputs whose connection the graph removed, with its source: made again once
    // the source changes.
    std::unordered_map<std::string, Port> rejected_auto_connections;

    void set_identifier(const std::string& new_identifier) {
        identifier = new_identifier;
        label = fmt::format("{} ({})", identifier, type_name);
    }

    void reset() {
        static_cast<NodeConnectState&>(*this) = {};
    }

    NodeState state() const {
        if (bypassed) {
            return NodeState::BYPASSED;
        }
        if (unsupported) {
            return NodeState::UNSUPPORTED;
        }
        if (!enabled) {
            return NodeState::DISABLED;
        }
        if (!errors.empty()) {
            return NodeState::ERRORED;
        }
        if (force_disabled) {
            return NodeState::FORCE_DISABLED;
        }
        return NodeState::OK;
    }

    // As far as known: errors and force disabling are found while connecting.
    bool runs() const {
        return state() == NodeState::OK;
    }

    std::string_view state_name() const {
        switch (state()) {
        case NodeState::DISABLED:
            return "DISABLED";
        case NodeState::UNSUPPORTED:
            return "UNSUPPORTED";
        case NodeState::BYPASSED:
            return "BYPASSED";
        case NodeState::FORCE_DISABLED:
            return "F-DISABLED";
        case NodeState::ERRORED:
            return "ERROR";
        default:
            return "OK";
        }
    }

    uint32_t set_index(const uint64_t run_iteration) const {
        assert(!resource_maps.empty());
        return run_iteration % resource_maps.size();
    }
};

} // namespace graph_internal
} // namespace merian
