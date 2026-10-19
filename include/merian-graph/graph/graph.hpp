#pragma once

#include "errors.hpp"
#include "graph_description.hpp"
#include "merian/utils/chrono.hpp"
#include "merian/utils/concurrent/thread_pool.hpp"
#include "merian/utils/ring_buffer.hpp"
#include "node.hpp"
#include "node_process_info.hpp"
#include "resource.hpp"
#include "time_provider.hpp"

#include "graph_data.hpp"
#include "graph_memory_report.hpp"
#include "graph_run.hpp"

#include "merian-graph/graph/node_registry.hpp"
#include "merian/shader/shader_compiler.hpp"
#include "merian/utils/math.hpp"
#include "merian/vk/command/submission.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/memory/memory_allocator_transient.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace merian {

struct GraphCreateInfo {
    const ContextHandle context;
    const ResourceAllocatorHandle resource_allocator;
};

class MerianGraphExtension;

enum class ProfilerReportKind : uint8_t {
    // since the last connect or profiler_restart_report()
    RUN,
    INTERVAL,
    BUILD,
};

/**
 * @brief      A Vulkan processing graph.
 */
class Graph : public std::enable_shared_from_this<Graph> {
    friend MerianGraphExtension;
    friend NodeIO;
    friend NodeIOLayout;

    using NodeData = graph_internal::NodeData;
    using ConnectorRef = graph_internal::ConnectorRef;
    using Connection = graph_internal::Connection;
    using InputDecision = graph_internal::InputDecision;
    using ResourceData = graph_internal::ResourceData;

    // Data that is stored for every iteration in flight.
    struct InFlightData {
        // The submission for the current iteration. One per slot so its command pool is only
        // reset once the slot's last iteration finished.
        SubmissionHandle submission;
        // Query pools for the profiler
        QueryPoolHandle<vk::QueryType::eTimestamp> profiler_query_pool;
        // reached by the iteration semaphore once the slot's last iteration finished
        uint64_t iteration_value = 0;
        // For each node: optional in-flight data.
        std::unordered_map<NodeHandle, std::any> in_flight_data{};
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

    template <typename NODE_TYPE>
    std::shared_ptr<NODE_TYPE> find_node_for_identifier_and_type(const std::string& identifier)
        requires(std::is_base_of_v<Node, NODE_TYPE>)
    {
        NodeHandle maybe_match = find_node_for_identifier(identifier);
        if (!maybe_match) {
            return nullptr;
        }
        if (node_data.at(maybe_match).type_name == registry.node_type_name<NODE_TYPE>()) {
            return debugable_ptr_cast<NODE_TYPE>(maybe_match);
        }

        return nullptr;
    }

    // Throws invalid_argument if one of the nodes does not exist. connect() removes the connection
    // if it cannot work. Replaces an existing connection to the same input. Children of a subgraph
    // cannot be connected; connect the public connectors of their parent.
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

    // Connects if necessary. stop_requested is polled after each after_iteration and while waiting
    // for the GPU, during which the host callbacks of finished iterations run.
    GraphRunResult run(const GraphRunOptions& options = {},
                       const std::function<void()>& after_iteration = {},
                       const std::function<bool()>& stop_requested = {});

    // Also runs the host callbacks and waits for the async ones. False on timeout.
    bool wait(const std::optional<std::chrono::nanoseconds>& timeout = {});

    void clear();

    // Reconnects before the next iteration: run iteration 0, node histories cleared, the time since
    // connect and the RUN report restarted.
    void restart_run();

    // Recompiles the shaders whose files changed, includes and imports alike, all if force. Every
    // connect reloads.
    void reload_shaders(const bool force = false);

    bool get_needs_reconnect() const;

    // Of the next iteration, since the graph was created. Information only, e.g. for file names.
    uint64_t get_total_iteration() const;

    // Of the next iteration, since the last connect; 0 while a reconnect is pending.
    uint64_t get_run_iteration() const;

    // The graph time of the last iteration.
    std::chrono::nanoseconds get_time() const;

    std::ranges::keys_view<std::ranges::ref_view<const std::map<std::string, NodeHandle>>>
    get_identifiers() const;

    std::vector<std::string> get_top_level_identifiers() const;

    // Enabled nodes that did not run after the last connect, with the reason.
    std::vector<std::pair<std::string, std::string>> get_inactive_nodes() const;

    // Throws invalid_argument if the graph does not contain the node, as do the methods below.
    NodeState node_get_state(const std::string& identifier) const;

    const std::string& node_get_type(const std::string& identifier) const;

    NodeConnectors node_get_connectors(const std::string& identifier) const;

    // Changes apply once the call returned.
    void node_properties(const std::string& identifier, Properties& props);

    nlohmann::json node_get_properties(const std::string& identifier) const;

    // Others keep their value. Throws std::invalid_argument naming unknown keys and invalid values
    // after setting the rest.
    void node_set_properties(const std::string& identifier, const nlohmann::json& properties);

    // --- Events ---

    void send_event(const std::string& event_name, const GraphEvent::Data& data = {});

    // --- Load/Store ---

    // Key/value description of this run for a node to embed into what it writes.
    std::vector<std::pair<std::string, std::string>> get_metadata();

    // Returns per node the properties it did not read or with invalid values.
    std::vector<std::string> from_description(const GraphDescription& description);

    // Create description from current graph state
    GraphDescription to_description();

    // --- Properties ---

    // "graph_properties" and "profiler"; changes apply once the call returned.
    void properties(Properties& props);

    nlohmann::json get_properties();

    // Others keep their value. Throws std::invalid_argument naming unknown keys and invalid values
    // after setting the rest.
    void set_properties(const nlohmann::json& properties);

    // --- Reports ---

    // Not stored. Off by default; while on, the nodes of a layer do not overlap on the GPU.
    void profiler_set_enabled(const bool enabled);

    bool profiler_get_enabled() const;

    // Of the iterations whose GPU timings were collected; wait() collects all.
    ProfilerReport profiler_get_report(const ProfilerReportKind kind) const;

    void profiler_restart_report();

    GraphMemoryReport memory_get_report() const;

    // --- UI ---

    void draw_ui(Properties& props);

    // The default path of the panel's Load and Store.
    void ui_set_store_path(const std::filesystem::path& path);

  private:
    /*--- Graph Edit --- */

    const std::string& add_node(const std::shared_ptr<Node>& node, const std::string& identifier);

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

    void request_reconnect();

    void connect_graph();
    bool run_unless_stopped(const std::function<bool()>& stop_requested);
    bool run_once(const std::function<bool()>& stop_requested);
    // Runs the finished host callbacks while waiting; false if stop_requested returned true.
    bool wait_for_iteration(const uint64_t value, const std::function<bool()>& stop_requested);
    // Oldest iteration first; returns the wait of the first pending callback.
    std::optional<std::pair<TimelineSemaphoreHandle, uint64_t>> run_finished_callbacks();
    // Runs the host callbacks that commands in flight wait for.
    void run_host_round_trips();

    // Of the in-flight slots, 0 the oldest.
    const SubmissionHandle& in_flight_submission(const uint32_t age);
    // Only while no iteration is in flight.
    void resize_in_flight(const uint32_t size);
    // waits while settings.max_async_tasks run and another one waits
    void run_async(std::function<void()>&& task);
    void record_iteration(Submission& submission, const ProfilerHandle& profiler);
    void finish_iteration(InFlightData& in_flight_data);

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

    void graph_settings(Properties& props);
    void profiler_settings(Properties& props);

    void node_section(Properties& props, const NodeHandle& node, const bool edit);

    void output_section(Properties& props, NodeData& data, const OutputConnectorHandle& output);
    void public_output_section(Properties& props, const ConnectorRef& source);
    void input_section(Properties& props, NodeData& data, const InputConnectorHandle& input);
    void public_input_section(Properties& props,
                              const NodeHandle& node,
                              const std::string& input,
                              const std::vector<ConnectorRef>& fed_inputs);

    void apply_properties_status(const std::string& identifier, Node::NodeStatusFlags flags);

    // Describes unknown keys, values of the wrong type and invalid values, if any.
    std::optional<std::string> load_node_properties(const std::string& identifier,
                                                    const NodeHandle& node,
                                                    const nlohmann::json& properties);

    void draw_controls(Properties& props);
    void draw_state(Properties& props);
    void draw_edit(Properties& props);
    void draw_profiler(Properties& props);
    void draw_memory(Properties& props);

    // --- Graph run sub-tasks ---

    void advance_time();

    // The time source continues from the held time.
    void release_time();

    // The time since connect moves along; the system clock continues from the new time.
    void jump_time(const std::chrono::nanoseconds time);

    // Resets the run iteration, the time since connect and the RUN report.
    void reset_run();

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

    static constexpr vk::MemoryBarrier2 FULL_MEMORY_BARRIER{
        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};

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
    static std::optional<ConnectorRef> auto_connection_source(
        const NodeHandle& node,
        const NodeData& data,
        std::string input,
        const std::function<std::optional<ConnectorRef>(const std::string&)>& source_of);

    // The desired or subgraph connection that feeds the input, followed through public inputs.
    std::optional<Connection> wiring(NodeHandle node, std::string input) const;

    // What consumers of the output receive: through public outputs, user-disabled nodes that pass
    // through and the versions of writers that do not run.
    ConnectorRef follow(ConnectorRef output) const;

    // The child output that carries a public output, through nested subgraphs.
    ConnectorRef carrier(ConnectorRef output) const;

    // wired if it is a desired connection that passes only public connectors and its source is not
    // an in-place version: removing it fixes the rejection.
    std::optional<Connection> direct_connection(ConnectorRef consumer,
                                                const Connection& wired,
                                                const ConnectorRef& source) const;

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
    // children. A parent with errors gets no connectors.
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

    // Until the next connect or the node's removal.
    void register_event_listener_for_connect(const NodeHandle& node,
                                             const std::vector<GraphEvent::Pattern>& patterns,
                                             const GraphEvent::Listener& event_listener);

    void warn_if_no_sender(const GraphEvent::Pattern& pattern) const;

    void send_graph_event(const std::string& event_name, const GraphEvent::Data& data = {});

    void send_event(const GraphEvent::Info& event_info, const GraphEvent::Data& data);

    // General stuff
    const ContextHandle context;
    const ResourceAllocatorHandle resource_allocator;
    const QueueHandle queue;
    std::shared_ptr<ExtensionVkDebugUtils> debug_utils = nullptr;

    std::mutex async_mutex;
    std::condition_variable async_finished;
    uint32_t async_running = 0;
    ThreadPoolHandle async_tasks;
    const Submission::AsyncRunner async_runner;

    NodeRegistry& registry;

    // Per-iteration data management
    std::vector<InFlightData> in_flight;
    uint32_t in_flight_index = 0;

    // State
    bool needs_reconnect = true;
    bool profiler_enable = false;
    uint32_t busy = 0;
    std::vector<std::function<void()>> deferred_edits;

    uint64_t total_iteration = 0;
    uint64_t run_iteration = 0;
    std::chrono::high_resolution_clock::time_point time_reference;
    std::chrono::high_resolution_clock::time_point time_connect_reference;
    std::chrono::nanoseconds duration_elapsed_since_connect{};
    std::chrono::nanoseconds duration_elapsed{};
    static constexpr const char* TIME_SOURCE_AUTO = "Auto";
    static constexpr const char* TIME_SOURCE_SYSTEM = "System";
    static constexpr const char* TIME_SOURCE_TIME = "Time";
    static constexpr const char* TIME_SOURCE_DELTA = "Delta";

    // What a description's graph_properties and profiler set; defaults again on each load.
    struct Settings {
        uint32_t iterations_in_flight = 2;
        uint32_t seed = 0;
        // Identifier of the node that drives the clock, or one of the TIME_SOURCE_* names.
        std::string time_source = TIME_SOURCE_AUTO;
        // Applied once per run for Delta, once and then reset for Time.
        float time_overwrite_delta_ms = 0.;
        int32_t limit_fps = 0;
        bool alias_transient_resources = true;
        uint32_t max_async_tasks = 8;
        std::string imgui_event = "ui";
        uint32_t profiler_report_interval_ms = 50;
        uint32_t profiler_evict_after_ms = 5000;
    };
    Settings settings;
    // the time of the last run that set one, until a run without
    std::optional<std::chrono::nanoseconds> held_time;
    // to the previous iteration, also across connects
    std::chrono::nanoseconds time_delta{};
    std::chrono::nanoseconds cpu_time{};

    // Reused by the next connect. (on plan_memory)
    std::vector<TransientMemoryAllocator::Block> transient_blocks;

    std::chrono::duration<double> gpu_wait_time{};
    std::chrono::duration<double> cpu_sleep_time{};

    ProfilerReport build_report;
    ProfilerReport interval_report;
    merian::ProfilerHandle run_profiler;

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
    struct EventListener {
        // the registering node, nullptr for the graph
        NodeHandle node;
        std::vector<GraphEvent::Pattern> patterns;
        GraphEvent::Listener listener;
    };
    std::vector<EventListener> event_listeners;

    // Set while run() runs; the first matching event sent ends it.
    std::vector<GraphEvent::Pattern> run_stop_patterns;
    std::optional<std::string> run_stop_event;

    // Most recently sent events for the properties readout, deduped by name.
    struct RecentEvent {
        uint64_t iteration;
        std::chrono::steady_clock::time_point time;
    };
    std::map<std::string, RecentEvent> recent_events;

    struct UiState {
        struct MemoryView {
            std::string summary;
            std::string nodes;
            std::string layers;
            std::string outputs;
            std::string blocks;
        };

        std::string store_path;
        std::string send_event;
        int new_node_selected = 0;
        std::string new_node_identifier;
        int remove_node_selected = 0;
        int add_connection_selected_src = 0;
        int add_connection_selected_src_output = 0;
        int add_connection_selected_dst = 0;
        int add_connection_selected_dst_input = 0;

        // in ms
        RingBuffer<float> cpu_time_history{256};
        RingBuffer<float> gpu_time_history{256};
        float cpu_max = 20, gpu_max = 20;
        bool cpu_auto = true, gpu_auto = true;
        uint32_t time_history_current = 0;

        // rebuilt after a connect and at most once per second
        MemoryView memory_view;
        bool memory_view_dirty = true;
        std::chrono::steady_clock::time_point memory_view_time;
    };
    UiState ui;

    NodeProcessInfo run_info;
    const TimelineSemaphoreHandle iteration_semaphore;

    std::shared_ptr<MerianGraphExtension> context_extension;
};

using GraphHandle = std::shared_ptr<Graph>;

} // namespace merian
