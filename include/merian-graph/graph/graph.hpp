#pragma once

#include "errors.hpp"
#include "graph_description.hpp"
#include "merian/utils/chrono.hpp"
#include "merian/utils/concurrent/thread_pool.hpp"
#include "merian/utils/ring_buffer.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/utils/cpu_queue.hpp"
#include "node.hpp"
#include "node_process_info.hpp"
#include "resource.hpp"
#include "time_provider.hpp"

#include "graph_data.hpp"
#include "graph_memory_report.hpp"

#include "merian-graph/graph/node_registry.hpp"
#include "merian/shader/shader_compiler.hpp"
#include "merian/utils/math.hpp"
#include "merian/vk/command/submission.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/memory/memory_allocator_transient.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/sync/ring_fences.hpp"

#include <cstdint>
#include <regex>
#include <unordered_map>
#include <unordered_set>

#include <fmt/chrono.h>

namespace merian {

using namespace std::literals::chrono_literals;

struct GraphCreateInfo {
    const ContextHandle context;
    const ResourceAllocatorHandle resource_allocator;
};

class MerianGraphExtension;

/**
 * @brief      A Vulkan processing graph.
 */
class Graph : public std::enable_shared_from_this<Graph> {
    friend MerianGraphExtension;
    friend NodeIO;
    friend NodeIOLayout;

    using NodeData = graph_internal::NodeData;
    using Port = graph_internal::Port;
    using Connection = graph_internal::Connection;
    using InputDecision = graph_internal::InputDecision;
    using ResourceData = graph_internal::ResourceData;

    // Data that is stored for every iteration in flight.
    // Created for each iteration in flight in Graph::Graph.
    struct InFlightData {
        // The submission for the current iteration. One per slot so its command pool is only
        // reset once the iteration's fence was awaited.
        SubmissionHandle submission;
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

    // Adds a node of the registered type, unsupported if it fails to initialize.
    //
    // Throws invalid_argument if a node with this identifier already exists or it is invalid.
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

    // Throws invalid_argument if one of the nodes does not exist. connect() removes the connection
    // if it cannot work. Replaces an existing connection to the same input. Children of a subgraph
    // cannot be connected; connect the public ports of their parent.
    void add_connection(const std::string& src,
                        const std::string& dst,
                        const std::string& src_output,
                        const std::string& dst_input);

    // False if src is not connected to the input.
    bool
    remove_connection(const std::string& src, const std::string& dst, const std::string& dst_input);

    // Removes the node with its children; children cannot be removed on their own. Deferred while
    // running, connecting or drawing properties.
    bool remove_node(const std::string& identifier);

    // --- connect / run graph ---

    // Attempts to connect the graph with the current set of connections.
    //
    // Connections that cannot work are removed. In this case connect returns with
    // needs_reconnect still being true. For this reason connect should be called in a loop.
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

    // Throws invalid_argument if the graph does not contain the node.
    NodeState get_node_state(const std::string& identifier) const;

    std::ranges::keys_view<std::ranges::ref_view<const std::map<std::string, NodeHandle>>>
    identifiers();

    // --- Events ---

    void send_event(const std::string& event_name,
                    const GraphEvent::Data& data = {},
                    const bool notify_all = true);

    void register_event_listener(const std::string& event_pattern,
                                 const GraphEvent::Listener& event_listener);

    // --- Load/Store ---

    // Load graph from JSON file or object
    void load_from_file(const std::filesystem::path& path);
    void load_from_json(const nlohmann::json& json);

    // Store graph to JSON file or object
    void store_to_file(const std::filesystem::path& path);
    nlohmann::json store_to_json();

    // Key/value description of this run for a node to embed into what it writes.
    std::vector<std::pair<std::string, std::string>> metadata();

    // Sets the default path for the load/store controls in the graph properties. load_from_file
    // sets this automatically; a host that loads via load_from_json should set it explicitly.
    void set_store_path(const std::filesystem::path& path);

    // Build graph from description
    void from_description(const GraphDescription& description);

    // Create description from current graph state
    GraphDescription to_description();

    // --- Time ---

    // Advances the graph time by a fixed delta per iteration instead of following the wall clock.
    void set_time_source_delta(const float delta_ms);

    // --- Profiling ---

    // Off by default. While on, the nodes of a layer do not overlap on the GPU.
    void set_profiler_enabled(const bool enabled);

    // Averages and deviations in the run report are computed over this period.
    void set_profiler_report_interval(const uint32_t millis);

    // Report over the period since the last one; call after the last run to summarize a benchmark.
    Profiler::Report get_run_report();

    // Report of the last connect.
    const Profiler::Report& get_build_report() const;

    // --- Memory ---

    GraphMemoryReport get_memory_report() const;

    // --- Properties / Graph UI ---

    void properties(Properties& props);

  private:
    /*--- Graph Edit --- */

    const std::string& add_node(const std::shared_ptr<Node>& node, const std::string& identifier);

    std::string unique_identifier(const std::string& base) const;

    // Throws invalid_argument if the graph does not contain it.
    NodeHandle node_for(const std::string& identifier) const;

    // Applies the edit now, or once no run, connect or properties() is in progress.
    void when_idle(std::function<void()> edit);
    void apply_deferred_edits();

    struct BusyScope {
        explicit BusyScope(Graph& graph) : graph(graph) {
            graph.busy++;
        }
        ~BusyScope() {
            graph.busy--;
        }
        BusyScope(const BusyScope&) = delete;
        BusyScope& operator=(const BusyScope&) = delete;

        Graph& graph;
    };

    void connect_graph();
    void run_once();
    void draw_properties(Properties& props);

    // Without validating the identifier or requesting a reconnect.
    void insert_node(const NodeHandle& node,
                     const std::string& identifier,
                     const NodeHandle& parent = nullptr);

    // Removes the node, its children and their connections.
    void erase_node(const NodeHandle& node);

    // Ordered by identifier.
    std::vector<NodeHandle> top_level_nodes() const;

    void add_connection(const NodeHandle& src,
                        const NodeHandle& dst,
                        const std::string& src_output,
                        const std::string& dst_input);

    // Without requesting a reconnect.
    void add_desired_connection(const NodeHandle& src,
                                const NodeHandle& dst,
                                const std::string& src_output,
                                const std::string& dst_input);

    bool
    remove_connection(const NodeHandle& src, const NodeHandle& dst, const std::string& dst_input);

    /*--- Properties ---*/
    void graph_properties(Properties& props);
    void profiler_properties(Properties& props);
    struct MemoryView {
        std::string summary;
        std::string nodes;
        std::string layers;
        std::string outputs;
        std::string blocks;
    };

    // Rebuilt after a connect and at most once per second.
    const MemoryView& memory_view_for_ui();

    void memory_properties(Properties& props);

    void node_properties(Properties& props, const NodeHandle& node);
    void io_props_for_node(Properties& config, const NodeHandle& node, NodeData& data);

    // --- Graph run sub-tasks ---

    void advance_time();

    // In topological order.
    std::vector<NodeHandle> time_providers() const;

    // Creates the profiler if necessary
    ProfilerHandle prepare_profiler_for_run(InFlightData& in_flight_data);

    // Collects the barriers of the node's connectors into image_barriers and buffer_barriers.
    void pre_process_connectors(Submission& submission, const NodeHandle& node, NodeData& data);
    void process_node(Submission& submission,
                      const NodeHandle& node,
                      NodeData& data,
                      const ProfilerHandle& profiler);
    void post_process_connectors(Submission& submission, const NodeHandle& node, NodeData& data);
    void flush_barriers(Submission& submission, const vk::MemoryBarrier2& memory_barrier = {});

    // --- connect ---

    template <typename F>
    static void collect_node_errors(std::vector<std::string>& errors, const F& f) {
        try {
            f();
        } catch (const graph_errors::connector_error& e) {
            errors.emplace_back(fmt::format("connector error: {}", e.what()));
        } catch (const graph_errors::node_error& e) {
            errors.emplace_back(fmt::format("node error: {}", e.what()));
        } catch (const ShaderCompiler::compilation_failed& e) {
            errors.emplace_back(fmt::format("compilation failed: {}", e.what()));
        }
    }

    // Frees the resources and forgets what the last connect made of the nodes.
    void reset_connections();

    // Connects the nodes; topology receives the nodes that run, in visit order. Returns false if
    // desired connections were removed or an in-place write was rejected and connect must restart.
    [[nodiscard]]
    bool connect_nodes(std::vector<NodeHandle>& topology);

    void cache_node_input_connectors();

    // Makes each input's auto connection once, unless it is connected otherwise; not for children.
    void materialize_auto_connections();

    // The source of the input's auto connection, following inputs that are auto connected
    // themselves; source_of resolves a connected input.
    static std::optional<Port>
    auto_connection_source(const NodeHandle& node,
                           const NodeData& data,
                           std::string input,
                           const std::function<std::optional<Port>(const std::string&)>& source_of);

    // The desired or subgraph connection that feeds the input, followed through public inputs.
    std::optional<Connection> wiring(NodeHandle node, std::string input) const;

    // What consumers of the output receive: through public outputs, user-disabled nodes that pass
    // through and the versions of writers that do not run.
    Port follow(Port port) const;

    // wired if it is a desired connection that passes only public ports and its source is not an
    // in-place version: removing it fixes the rejection.
    std::optional<Connection>
    direct_connection(Port consumer, const Connection& wired, const Port& source) const;

    InputDecision decide_input(const NodeHandle& node,
                               const NodeData& data,
                               const InputConnectorHandle& input) const;

    // Connects the input, leaves an optional one unconnected or stops the node. Removable
    // rejections are added to removals. Returns whether the input's connection changed.
    bool apply_decision(NodeData& data,
                        const InputConnectorHandle& input,
                        const InputDecision& decision,
                        std::vector<Connection>& removals);

    // Visits the nodes in identifier order once their undelayed inputs are decided and describes
    // their outputs.
    void visit_nodes(std::vector<NodeHandle>& topology, std::vector<Connection>& removals);

    // The nodes on a cycle of nodes waiting for each other's outputs. (node -> producers)
    std::vector<NodeHandle> waiting_on_themselves(
        const std::unordered_map<NodeHandle, std::vector<NodeHandle>>& waiting) const;

    // Decides the delayed inputs and those whose producer stopped later, until stable, and drops
    // the nodes that stopped.
    void settle(std::vector<NodeHandle>& topology, std::vector<Connection>& removals);

    // Records the inputs each output feeds.
    void connect_outputs(const std::vector<NodeHandle>& topology);

    // Keeps the first reason down a chain.
    void force_disable(NodeData& data, const std::string& reason);

    // Force disables a running child whose parent does not run; true if it did.
    bool stop_with_parent(NodeData& data);

    // Why the node does not run, as far as known, empty if it runs.
    std::string not_running_reason(const NodeHandle& node) const;

    // describe_outputs, plus an output for each in-place input.
    void cache_node_output_connectors(const NodeHandle& node, NodeData& data);

    // A writer runs after the readers of its version, at most one node writes a version, no one
    // reads it delayed. Queues errors on the nodes that break this and returns false.
    [[nodiscard]]
    bool validate_in_place_writes(const std::vector<NodeHandle>& nodes);

    std::string make_error_input_not_connected(const InputConnectorHandle& input,
                                               const NodeData& data);

    // --- subgraphs ---

    // Calls describe_subgraph() of every node, parents before their children, and adds, removes,
    // moves and renames the children accordingly.
    void build_subgraphs();

    // None for an unsupported node, and for one failing to describe its subgraph, which errors it.
    std::optional<Subgraph> subgraph_of(const NodeHandle& node);

    // (child -> the nodes that describe it)
    void build_subgraph(const NodeHandle& node,
                        const std::optional<Subgraph>& subgraph,
                        const std::unordered_map<NodeHandle, std::vector<NodeHandle>>& describers);

    // Validates each subgraph, parents before their children, and connects the inputs of the
    // children. A parent with errors gets no ports.
    void connect_subgraphs();

    void connect_subgraph(NodeData& data);

    // --- layers and resources ---

    // The nodes each node must run after: the producers of its undelayed inputs and, for each
    // version it writes in place, the other readers of that version.
    std::unordered_map<NodeHandle, std::vector<NodeHandle>>
    compute_predecessors(const std::vector<NodeHandle>& nodes) const;

    void build_layers(const std::vector<NodeHandle>& nodes);

    // Creates the copies of every output, places their memory and adds the discards and barriers
    // where memory changes hands.
    void plan_memory();

    void precompute_resources();

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
    void
    set_on_run_starting(const std::function<void(const NodeProcessInfo& info)>& on_run_starting);

    // Set a callback that is executed right before the commands for this run are submitted to
    // the GPU.
    void set_on_pre_submit(const std::function<void(const NodeProcessInfo& info)>& on_pre_submit);

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
    std::function<void(const NodeProcessInfo&)> on_run_starting = [](const NodeProcessInfo&) {};
    std::function<void(const NodeProcessInfo&)> on_pre_submit = [](const NodeProcessInfo&) {};
    std::function<void()> on_post_submit = [] {};

    // Per-iteration data management
    uint32_t desired_iterations_in_flight = 2;
    // set to desired_iterations_in_flight in connect()
    merian::RingFences<InFlightData> ring_fences;

    // State
    bool needs_reconnect = false;
    bool profiler_enable = false;
    uint32_t profiler_report_intervall_ms = 50;
    uint32_t profiler_evict_after_ms = 5000;
    uint32_t busy = 0;
    std::vector<std::function<void()>> deferred_edits;

    uint64_t total_iteration = 0;
    uint64_t run_iteration = 0;
    std::chrono::high_resolution_clock::time_point time_reference;
    std::chrono::high_resolution_clock::time_point time_connect_reference;
    std::chrono::nanoseconds duration_elapsed_since_connect;
    std::chrono::nanoseconds duration_elapsed;
    static constexpr const char* TIME_SOURCE_AUTO = "Auto";
    static constexpr const char* TIME_SOURCE_SYSTEM = "System";
    static constexpr const char* TIME_SOURCE_TIME = "Time";
    static constexpr const char* TIME_SOURCE_DELTA = "Delta";
    // Identifier of the node that drives the clock, or one of the TIME_SOURCE_* names.
    std::string time_source = TIME_SOURCE_AUTO;
    // Applied once per run for Delta, once and then reset for Time.
    float time_overwrite_delta_ms = 0.;
    // across builds. Might be not 0 at begin of run.
    std::chrono::nanoseconds time_delta = 0ns;
    std::chrono::nanoseconds cpu_time = 0ns;

    bool flush_thread_pool_at_run_start = true;

    bool alias_transient_resources = true;
    // Reused by the next connect. (on plan_memory)
    std::vector<TransientMemoryAllocator::Block> transient_blocks;

    bool memory_view_dirty = true;
    MemoryView memory_view;
    std::chrono::steady_clock::time_point memory_view_time;

    std::chrono::duration<double> gpu_wait_time = 0ns;
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

    // Path of the Load and Store controls.
    std::string store_path;
    std::string imgui_event = "ui";

    GraphDescription loaded_description;

    // Nodes
    std::map<std::string, NodeHandle> node_for_identifier;
    std::unordered_map<NodeHandle, NodeData> node_data;

    // Nodes of one dependency layer have no ordering constraints among each other; the barrier
    // is emitted before the layer's nodes run. Its src covers all previous layers cumulatively,
    // all commands where memory changes hands; the barrier of layer 0 orders the whole iteration
    // against the previous one.
    struct Layer {
        struct Set {
            vk::MemoryBarrier2 barrier;
            // images whose memory another resource used since their last use
            std::vector<ImageHandle> discards;
        };
        // (iteration % period) is the set of an iteration
        std::vector<Set> sets;
        std::vector<NodeHandle> nodes;
    };
    // The topology in execution order. (on build_layers)
    std::vector<Layer> layers;
    // lcm(delay + 1) over the delayed inputs: the iterations after which the resources repeat.
    // (on build_layers)
    uint32_t period = 1;
    // The connector barriers of a layer.
    std::vector<vk::ImageMemoryBarrier2> image_barriers;
    std::vector<vk::BufferMemoryBarrier2> buffer_barriers;

    std::shared_ptr<FrameCachingShaderObjectAllocator> shader_object_allocator;

    // Events
    // (NodeHandle == nullptr means user events, event with name "" means "any")
    std::map<std::string, std::map<std::string, std::vector<GraphEvent::Listener>>> event_listeners;
    inline static const std::regex EVENT_REGEX{"([^/]*)/([^/]*)/([^/]*)"};

    // cached here when the user calls register_event_listener and added to the data structure
    // above when the graph is built.
    std::vector<std::pair<std::string, GraphEvent::Listener>> user_event_pattern_listener;

    // Most recently sent events for the properties readout, deduped by name. Updated in
    // send_event(), so it is mutable to keep that method const.
    struct RecentEvent {
        uint64_t iteration;
        std::chrono::steady_clock::time_point time;
    };
    mutable std::map<std::string, RecentEvent> recent_events;

    // Properties helper
    std::string props_send_event;
    int new_node_selected = 0;
    std::string new_node_identifier;
    int remove_node_selected = 0;
    int add_connection_selected_src = 0;
    int add_connection_selected_src_output = 0;
    int add_connection_selected_dst = 0;
    int add_connection_selected_dst_input = 0;

    NodeProcessInfo run_info;
    TimelineSemaphoreHandle iteration_semaphore;

    std::shared_ptr<MerianGraphExtension> context_extension;
};

using GraphHandle = std::shared_ptr<Graph>;

} // namespace merian
