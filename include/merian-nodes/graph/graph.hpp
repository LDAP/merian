#pragma once

#include "errors.hpp"
#include "graph_run.hpp"
#include "merian/utils/chrono.hpp"
#include "merian/utils/concurrent/thread_pool.hpp"
#include "merian/utils/ring_buffer.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/utils/cpu_queue.hpp"
#include "node.hpp"
#include "resource.hpp"

#include "graph_data.hpp"

#include "merian-nodes/graph/node_registry.hpp"
#include "merian/shader/shader_compiler.hpp"
#include "merian/utils/math.hpp"
#include "merian/vk/command/caching_command_pool.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/sync/ring_fences.hpp"

#include <cstdint>
#include <queue>
#include <regex>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <fmt/chrono.h>

namespace merian {

using namespace merian;
using namespace graph_internal;
using namespace std::literals::chrono_literals;

struct GraphCreateInfo {
    const ContextHandle context;
    const ResourceAllocatorHandle resource_allocator;
};

/**
 * @brief      A Vulkan processing graph.
 *
 * Implementation is split across multiple files:
 * - graph.cpp: Core execution (constructor, destructor, run, wait, reset)
 * - graph_edit.cpp: Node/connection management (add/remove nodes/connections)
 * - graph_connect.cpp: Graph topology and connection algorithm
 * - graph_properties.cpp: UI and serialization
 * - graph_events.cpp: Event system
 */
class Graph : public std::enable_shared_from_this<Graph> {
    friend class MerianNodesExtension;

    // Data that is stored for every iteration in flight.
    // Created for each iteration in flight in Graph::Graph.
    struct InFlightData {
        // The command pool for the current iteration.
        // We do not use RingCommandPool here since we might want to add a more custom
        // setup later (multi-threaded, multi-queues,...).
        CommandPoolHandle command_pool;
        std::shared_ptr<CachingCommandPool> command_buffer_cache;
        // Query pools for the profiler
        QueryPoolHandle<vk::QueryType::eTimestamp> profiler_query_pool;
        // Tasks that should be run in the current iteration after acquiring the fence.
        std::vector<std::function<void()>> tasks;
        // For each node: optional in-flight data.
        std::unordered_map<NodeHandle, std::any> in_flight_data{};
        // How long did the CPU delay processing
        std::chrono::duration<double> cpu_sleep_time = 0ns;
    };

  private:
    Graph(const GraphCreateInfo& create_info);

  public:
    ~Graph();

    // --- add / remove nodes and connections ---

    // Adds a node to the graph.
    //
    // The node_name must be a known type to the registry.
    //
    // Throws invalid_argument, if a node with this identifier already exists.
    //
    // Returns the node identifier.
    const std::string& add_node(const std::string& node_name,
                                const std::optional<std::string>& identifier = std::nullopt);

    // Returns nullptr if the node does not exist.
    NodeHandle find_node_for_identifier(const std::string& identifier) const;

    // finds any node with the given type. Returns nullptr if not found.
    template <typename NODE_TYPE>
    std::shared_ptr<NODE_TYPE> find_node_for_type()
        requires(std::is_base_of_v<Node, NODE_TYPE>)
    {
        for (const auto& [node, data] : node_data) {
            if (registry.node_type_name(node) == registry.node_type_name<NODE_TYPE>()) {
                return debugable_ptr_cast<NODE_TYPE>(node);
            }
        }

        return nullptr;
    }

    template <typename NODE_TYPE>
    std::shared_ptr<NODE_TYPE> find_node_for_identifier_and_type(const std::string& identifier)
        requires(std::is_base_of_v<Node, NODE_TYPE>)
    {
        NodeHandle maybe_match = find_node_for_identifier(identifier);
        if (!maybe_match) {
            return nullptr;
        }
        if (registry.node_type_name(maybe_match) == registry.node_type_name<NODE_TYPE>()) {
            return debugable_ptr_cast<NODE_TYPE>(maybe_match);
        }

        return nullptr;
    }

    // Adds a connection to the graph.
    //
    // Throws invalid_argument if one of the node does not exist in the graph.
    // The connection is validated on connect(). This means if you want to validate the connection
    // make sure to call connect() as well.
    //
    // New conenctions replace existing connections to the same input.
    void add_connection(const std::string& src,
                        const std::string& dst,
                        const std::string& src_output,
                        const std::string& dst_input);

    bool
    remove_connection(const std::string& src, const std::string& dst, const std::string& dst_input);

    // Removes a node from the graph.
    //
    // If a run is in progress the removal is queued for the end of the run.
    bool remove_node(const std::string& identifier);

    // --- connect / run graph ---

    // Attempts to connect the graph with the current set of connections.
    //
    // Invalid connections are automatically eliminated. In this case connect returns with
    // needs_reconnect still being true. For this reason connect should be called in a loop.
    //
    // May fail with conenector_error if two input or output connectors have the same name.
    void connect();

    // Runs one iteration of the graph.
    //
    // If necessary, the graph is automatically connected.
    //
    // The execution is blocked until the fence according to the current iteration is signaled.
    // Interaction with the run is possible using the callbacks.
    void run();

    // waits until all in-flight iterations have finished
    void wait();

    // removes all nodes and connections from the graph.
    void reset();

    // Ensures at reconnect at the next run
    void request_reconnect();

    bool get_needs_reconnect() const;

    std::ranges::keys_view<std::ranges::ref_view<const std::map<std::string, NodeHandle>>>
    identifiers();

    // --- Events ---

    void send_event(const std::string& event_name,
                    const GraphEvent::Data& data = {},
                    const bool notify_all = true);

    void register_event_listener(const std::string& event_pattern,
                                 const GraphEvent::Listener& event_listener);

    // --- Properties / Graph (de)serialize ---

    void properties(Properties& props);

  private:
    /*--- Graph Edit --- */

    const std::string& add_node(const std::shared_ptr<Node>& node,
                                const std::optional<std::string>& identifier = std::nullopt);

    void add_connection(const NodeHandle& src,
                        const NodeHandle& dst,
                        const std::string& src_output,
                        const std::string& dst_input);

    bool remove_connection(const NodeHandle src, const NodeHandle dst, const std::string dst_input);

    /*--- Properties ---*/
    void io_props_for_node(Properties& config, NodeHandle& node, NodeData& data);

    // --- Graph run sub-tasks ---

    // Creates the profiler if necessary
    ProfilerHandle prepare_profiler_for_run(InFlightData& in_flight_data);

    // Calls connector callbacks, checks resource states and records as well as applies descriptor
    // set updates.
    void run_node(GraphRun& run,
                  const NodeHandle& node,
                  NodeData& data,
                  [[maybe_unused]] const ProfilerHandle& profiler);

    void record_descriptor_updates(NodeData& src_data,
                                   const OutputConnectorHandle& src_output,
                                   NodeData::PerOutputInfo& per_output_info,
                                   const uint32_t resource_index);

    // --- Graph connect sub-tasks ---

    // Removes all connections, frees graph resources and resets the precomputed topology.
    // Only keeps desired connections.
    void reset_connections();

    // Calls the describe_inputs() methods of the nodes and caches the result in
    // - node_data[].input_connectors
    // - node_data[].input_connector_for_name
    // - maybe_connected_inputs
    //
    // For all desired_connections: Ensures that the inputs exists (outputs are NOT checked)
    // and that the connections to all inputs are unique.
    [[nodiscard]]
    bool cache_node_input_connectors();

    // Only for a "satisfied node". Means, all inputs are connected, or delayed or optional and will
    // not be connected.
    void cache_node_output_connectors(const NodeHandle& node, NodeData& data);

    [[nodiscard]]
    bool connect_node(const NodeHandle& node,
                      NodeData& data,
                      const std::unordered_set<NodeHandle>& visited);

    // Helper for topological visit that calculates the next topological layer from the 'not yet
    // visited' cadidate nodes.
    //
    // Sets errors if a required non-delayed input is not connected. In this case the node
    // is removed from candidates.
    //
    // Satisfied means:
    // - All non-delayed optional inputs that might get connected are connected
    // - All non-delayed required inputs are connected
    // - Delayed inputs might be connected (but don't have to). Meaning: We also checkout a node
    // that does not run eventually. We do this to get the outputs for the GUI.
    //
    // This is used to initialize a topological traversal of the graph to connect the nodes.
    void search_satisfied_nodes(std::set<NodeHandle>& candidates,
                                std::priority_queue<NodeHandle>& queue);

    // Attemps to connect the graph from the desired connections.
    // Returns a topological order in which the nodes can be executed, which only includes
    // non-disabled nodes.
    //
    // Returns false if failed and needs reconnect.
    bool connect_nodes();

    void allocate_resources();

    void prepare_descriptor_sets();

    std::string make_error_input_not_connected(const InputConnectorHandle& input,
                                               const NodeHandle& node,
                                               const NodeData& data);

    void register_event_listener_for_connect(const std::string& event_pattern,
                                             const GraphEvent::Listener& event_listener);

    void send_graph_event(const std::string& event_name,
                          const GraphEvent::Data& data = {},
                          const bool notify_all = true);

    void send_event(const GraphEvent::Info& event_info,
                    const GraphEvent::Data& data,
                    const bool notify_all) const;

  public:
    // --- Callback setter ---

    // Set a callback that is executed right after nodes are preprocessed and before any node is
    // run.
    void set_on_run_starting(const std::function<void(GraphRun& graph_run)>& on_run_starting);

    // Set a callback that is executed right before the commands for this run are submitted to
    // the GPU.
    void set_on_pre_submit(const std::function<void(GraphRun& graph_run)>& on_pre_submit);

    // Set a callback that is executed right after the run was submitted to the queue and the
    // run callbacks were called.
    void set_on_post_submit(const std::function<void()>& on_post_submit);

  private:
    // General stuff
    const ContextHandle context;
    const ResourceAllocatorHandle resource_allocator;
    const QueueHandle queue;
    std::shared_ptr<ExtensionVkDebugUtils> debug_utils = nullptr;

    ThreadPoolHandle thread_pool;
    CPUQueueHandle cpu_queue;

    NodeRegistry& registry;

    // Outside callbacks
    // clang-format off
    std::function<void(GraphRun& graph_run)>                                on_run_starting = [](GraphRun&) {};
    std::function<void(GraphRun& graph_run)>                                on_pre_submit = [](GraphRun&) {};
    std::function<void()>                                                   on_post_submit = [] {};
    // clang-format on

    // Per-iteration data management
    uint32_t desired_iterations_in_flight = 2;
    // set to desired_iterations_in_flight in connect()
    merian::RingFences<InFlightData> ring_fences;

    // State
    bool needs_reconnect = false;
    bool profiler_enable = true;
    uint32_t profiler_report_intervall_ms = 50;
    bool run_in_progress = false;
    std::vector<std::function<void()>> on_run_finished_tasks;

    uint64_t total_iteration = 0;
    uint64_t run_iteration = 0;
    // assert(overwrite_time || elapsed == now() - time_reference)
    // to prevent divergence
    std::chrono::high_resolution_clock::time_point time_reference;
    std::chrono::high_resolution_clock::time_point time_connect_reference;
    std::chrono::nanoseconds duration_elapsed_since_connect;
    std::chrono::nanoseconds duration_elapsed;
    int time_overwrite = 0; // NONE, TIME, DIFFERENCE
    // this is also used for overwrite time. In this case this should only be applied once and
    // then reset.
    float time_delta_overwrite_ms = 0.;
    // across builds. Might be not 0 at begin of run.
    std::chrono::nanoseconds time_delta = 0ns;
    std::chrono::nanoseconds cpu_time = 0ns;

    bool flush_thread_pool_at_run_start = true;

    bool low_latency_mode = false;
    std::chrono::duration<double> gpu_wait_time = 0ns;
    std::chrono::duration<double> external_wait_time = 0ns;
    int32_t limit_fps = 0;

    Profiler::Report last_build_report;
    Profiler::Report last_run_report;
    // in ms
    RingBuffer<float> cpu_time_history{256};
    RingBuffer<float> gpu_time_history{256};
    float cpu_max = 20, gpu_max = 20;
    bool cpu_auto = true, gpu_auto = true;
    // Always write at cpu_time_history_current and cpu_time_history_current + (size >> 1)
    uint32_t time_history_current = 0;
    merian::ProfilerHandle run_profiler;

    // Nodes
    std::map<std::string, NodeHandle> node_for_identifier;
    std::unordered_map<NodeHandle, NodeData> node_data;
    // After connect() contains the nodes as far as a connection was possible in topological
    // order
    std::vector<NodeHandle> flat_topology;
    // Store connectors that might be connected in start_nodes.
    // There may still be an invalid connection or an outputing node might be actually disabled.
    std::unordered_map<InputConnectorHandle, NodeHandle> maybe_connected_inputs;

    // Events
    // (NodeHandle == nullptr means user events, event with name "" means "any")
    std::map<std::string, std::map<std::string, std::vector<GraphEvent::Listener>>> event_listeners;
    inline static const std::regex EVENT_REGEX{"([^/]*)/([^/]*)/([^/]*)"};

    // cached here when the user calls register_event_listener and added to the data structure
    // above when the graph is built.
    std::vector<std::pair<std::string, GraphEvent::Listener>> user_event_pattern_listener;

    // Properties helper
    std::string props_send_event;
    int new_node_selected = 0;
    std::string new_node_identifier;
    int remove_node_selected = 0;
    int add_connection_selected_src = 0;
    int add_connection_selected_src_output = 0;
    int add_connection_selected_dst = 0;
    int add_connection_selected_dst_input = 0;

    GraphRun graph_run;
};

using GraphHandle = std::shared_ptr<Graph>;

} // namespace merian
