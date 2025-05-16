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

#include "merian-nodes/graph/node_registry.hpp"
#include "merian/vk/command/caching_command_pool.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/sync/ring_fences.hpp"
#include "merian/vk/utils/math.hpp"

#include <cstdint>
#include <queue>
#include <regex>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <fmt/chrono.h>
namespace merian_nodes {
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
    static const uint32_t NO_DESCRIPTOR_BINDING = -1u;

    NodeData(const std::string& identifier) : identifier(identifier) {}

    // A unique name that identifies this node (user configurable).
    // This is not the name from the node registry.
    // (on add_node)
    std::string identifier;

    // User disabled
    bool disable{};
    // Errors during build
    std::vector<std::string> errors{};
    // Errors during run - triggers rebuild and gets build error
    std::vector<std::string> run_errors{};

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

        uint32_t descriptor_set_binding{NO_DESCRIPTOR_BINDING}; // (on prepare_descriptor_sets)
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
        uint32_t descriptor_set_binding{NO_DESCRIPTOR_BINDING}; // (on prepare_descriptor_sets)
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

    DescriptorPoolHandle descriptor_pool;

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
        descriptor_pool.reset();
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

using namespace merian;
using namespace graph_internal;
using namespace std::literals::chrono_literals;

/**
 * @brief      A Vulkan processing graph.
 *
 * @tparam     RING_SIZE  Controls the amount of in-flight processing (frames-in-flight).
 */
template <uint32_t ITERATIONS_IN_FLIGHT = 2>
class Graph : public std::enable_shared_from_this<Graph<ITERATIONS_IN_FLIGHT>> {
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

  public:
    Graph(const ContextHandle& context, const ResourceAllocatorHandle& resource_allocator)
        : context(context), resource_allocator(resource_allocator), queue(context->get_queue_GCT()),
          registry(context, resource_allocator), ring_fences(context),
          thread_pool(std::make_shared<ThreadPool>()),
          cpu_queue(std::make_shared<CPUQueue>(context, thread_pool)),
          run_profiler(std::make_shared<merian::Profiler>(context)),
          graph_run(ITERATIONS_IN_FLIGHT,
                    thread_pool,
                    cpu_queue,
                    run_profiler,
                    resource_allocator,
                    queue,
                    ShaderCompiler::get(context)) {

        for (uint32_t i = 0; i < ITERATIONS_IN_FLIGHT; i++) {
            InFlightData& in_flight_data = ring_fences.get(i).user_data;
            in_flight_data.command_pool = std::make_shared<CommandPool>(queue);
            in_flight_data.command_buffer_cache =
                std::make_shared<CachingCommandPool>(in_flight_data.command_pool);
            in_flight_data.profiler_query_pool =
                std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, 512, true);
        }

        debug_utils = context->get_extension<ExtensionVkDebugUtils>();
        time_connect_reference = time_reference = std::chrono::high_resolution_clock::now();
        duration_elapsed = 0ns;
    }

    ~Graph() {
        wait();
    }

    // --- add / remove nodes and connections ---

    NodeRegistry& get_registry() {
        return registry;
    }

    // Adds a node to the graph.
    //
    // The node_type must be a known type to the registry.
    //
    // Throws invalid_argument, if a node with this identifier already exists.
    //
    // Returns the node identifier.
    const std::string& add_node(const std::string& node_type,
                                const std::optional<std::string>& identifier = std::nullopt) {
        return add_node(registry.create_node_from_name(node_type), identifier);
    }

    // Returns nullptr if the node does not exist.
    NodeHandle find_node_for_identifier(const std::string& identifier) const {
        if (!node_for_identifier.contains(identifier)) {
            return nullptr;
        }
        return node_for_identifier.at(identifier);
    }

    // finds any node with the given type. Returns nullptr if not found.
    template <typename NODE_TYPE> std::shared_ptr<NODE_TYPE> find_node_for_type() {
        for (const auto& [node, data] : node_data) {
            if (registry.node_name(node) == registry.node_name<NODE_TYPE>()) {
                return debugable_ptr_cast<NODE_TYPE>(node);
            }
        }

        return nullptr;
    }

    template <typename NODE_TYPE>
    std::shared_ptr<NODE_TYPE> find_node_for_identifier_and_type(const std::string& identifier) {
        NodeHandle maybe_match = find_node_for_identifier(identifier);
        if (!maybe_match) {
            return nullptr;
        }
        if (registry.node_name(maybe_match) == registry.node_name<NODE_TYPE>()) {
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
                        const std::string& dst_input) {
        const NodeHandle src_node = find_node_for_identifier(src);
        const NodeHandle dst_node = find_node_for_identifier(dst);
        assert(src_node);
        assert(dst_node);
        add_connection(src_node, dst_node, src_output, dst_input);
    }

    bool remove_connection(const std::string& src,
                           const std::string& dst,
                           const std::string& dst_input) {
        const NodeHandle src_node = find_node_for_identifier(src);
        const NodeHandle dst_node = find_node_for_identifier(dst);
        assert(src_node);
        assert(dst_node);
        return remove_connection(src_node, dst_node, dst_input);
    }

    // Removes a node from the graph.
    //
    // If a run is in progress the removal is queued for the end of the run.
    bool remove_node(const std::string& identifier) {
        if (!node_for_identifier.contains(identifier)) {
            return false;
        }

        std::function<void()> remove_task = [this, identifier] {
            wait();

            const NodeHandle node = node_for_identifier.at(identifier);
            const NodeData& data = node_data.at(node);

            for (auto it = data.desired_outgoing_connections.begin();
                 it != data.desired_outgoing_connections.end();
                 it = data.desired_outgoing_connections.begin()) {
                remove_connection(node, it->dst, it->dst_input);
            }

            for (auto it = data.desired_incoming_connections.begin();
                 it != data.desired_incoming_connections.end();
                 it = data.desired_incoming_connections.begin()) {
                remove_connection(it->second.first, node, it->first);
            }

            const std::string node_identifier = data.identifier;
            node_data.erase(node);
            node_for_identifier.erase(identifier);
            for (uint32_t i = 0; i < ITERATIONS_IN_FLIGHT; i++) {
                InFlightData& in_flight_data = ring_fences.get(i).user_data;
                in_flight_data.in_flight_data.erase(node);
            }

            SPDLOG_DEBUG("removed node {} ({})", node_identifier, registry.node_name(node));
            needs_reconnect = true;
        };

        if (run_in_progress) {
            SPDLOG_DEBUG("schedule removal of node {} for the end of run the current run.",
                         identifier);
            on_run_finished_tasks.emplace_back(std::move(remove_task));
        } else {
            remove_task();
        }

        return true;
    }

    // --- connect / run graph ---

    // Attempts to connect the graph with the current set of connections.
    //
    // Invalid connections are automatically eliminated. In this case connect returns with
    // needs_reconnect still being true. For this reason connect should be called in a loop.
    //
    // May fail with conenector_error if two input or output connectors have the same name.
    void connect() {
        ProfilerHandle profiler = std::make_shared<Profiler>(context);
        {
            MERIAN_PROFILE_SCOPE(profiler, "connect");

            needs_reconnect = false;

            // no nodes -> no connect necessary
            if (node_data.empty()) {
                return;
            }

            {
                // let current nodes know that the graph is about to be reconnected.
                MERIAN_PROFILE_SCOPE(profiler, "notify nodes");
                send_graph_event("connect");
            }

            // Make sure resources are not in use
            {
                MERIAN_PROFILE_SCOPE(profiler, "wait for in-flight iterations");
                wait();
            }

            {
                MERIAN_PROFILE_SCOPE(profiler, "reset");
                reset_connections();
            }

            {
                MERIAN_PROFILE_SCOPE(profiler, "connect nodes");
                /*
                 * The connetion procedure works roughtly as follows:
                 * - while not all nodes were visited
                 *      - check if nodes must be disabled (required inputs cannot be satisfied)
                 *      - search nodes that are satisfied
                 *      - connect those nodes outputs with inputs
                 * - check if nodes must be disabled because of dependencies on backward edges are
                 * not satisfied
                 * - cleanup output connections to disabled nodes
                 * - call on_connect callbacks on the connectors
                 */
                if (!connect_nodes()) {
                    SPDLOG_WARN(
                        "Connecting nodes failed :( But attempted self healing. Retry, please!");
                    needs_reconnect = true;
                    return;
                }
            }

            {
                MERIAN_PROFILE_SCOPE(profiler, "allocate resources");
                allocate_resources();
            }

            {
                MERIAN_PROFILE_SCOPE(profiler, "prepare descriptor sets");
                prepare_descriptor_sets();
            }

            {
                MERIAN_PROFILE_SCOPE(profiler, "Node::on_connected");
                for (auto& node : flat_topology) {
                    NodeData& data = node_data.at(node);
                    MERIAN_PROFILE_SCOPE(profiler, fmt::format("{} ({})", data.identifier,
                                                               registry.node_name(node)));
                    SPDLOG_DEBUG("on_connected node: {} ({})", data.identifier,
                                 registry.node_name(node));
                    const NodeIOLayout io_layout(
                        [&](const InputConnectorHandle& input) {
#ifndef NDEBUG
                            if (std::find(data.input_connectors.begin(),
                                          data.input_connectors.end(),
                                          input) == data.input_connectors.end()) {
                                throw std::runtime_error{fmt::format(
                                    "Node {} tried to get an output connector for an input {} "
                                    "which was not returned in describe_inputs (which is not "
                                    "how this works).",
                                    registry.node_name(node), input->name)};
                            }
#endif
                            // for optional inputs we inserted a input connection with nullptr in
                            // search_satisfied_nodes, no problem here.
                            return data.input_connections.at(input).output;
                        },
                        [&](const std::string& event_pattern,
                            const GraphEvent::Listener& listener) {
                            register_event_listener_for_connect(event_pattern, listener);
                        });
                    const Node::NodeStatusFlags flags =
                        node->on_connected(io_layout, data.descriptor_set_layout);
                    needs_reconnect |= flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT;
                    if ((flags & Node::NodeStatusFlagBits::RESET_IN_FLIGHT_DATA) != 0u) {
                        for (uint32_t i = 0; i < ITERATIONS_IN_FLIGHT; i++) {
                            ring_fences.get(i).user_data.in_flight_data.at(node).reset();
                        }
                    }
                }
            }
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "register user event listener");
            for (const auto& [event_pattern, event_listener] : user_event_pattern_listener) {
                register_event_listener_for_connect(event_pattern, event_listener);
            }
        }

        run_iteration = 0;
        last_build_report = profiler->get_report();
        time_connect_reference = std::chrono::high_resolution_clock::now();
        duration_elapsed_since_connect = 0ns;
    }

    // Runs one iteration of the graph.
    //
    // If necessary, the graph is automatically connected.
    //
    // The execution is blocked until the fence according to the current iteration is signaled.
    // Interaction with the run is possible using the callbacks.
    void run() {
        // PREPARE RUN: wait for fence, release resources, reset cmd pool
        run_in_progress = true;

        if (flush_thread_pool_at_run_start) {
            thread_pool->wait_empty();
        }

        // wait for the in-flight processing to finish
        Stopwatch sw_gpu_wait;
        InFlightData& in_flight_data = ring_fences.next_cycle_wait_get();
        gpu_wait_time = gpu_wait_time * 0.9 + sw_gpu_wait.duration() * 0.1;

        // LOW LATENCY MODE
        if (low_latency_mode && !needs_reconnect) {
            const auto total_wait = std::max((std::max(gpu_wait_time, external_wait_time) +
                                              in_flight_data.cpu_sleep_time - 0.1ms),
                                             0.00ms);
            in_flight_data.cpu_sleep_time = 0.92 * total_wait;
        } else {
            in_flight_data.cpu_sleep_time = 0ms;
        }

        // FPS LIMITER
        if (limit_fps != 0) {
            in_flight_data.cpu_sleep_time =
                std::max(in_flight_data.cpu_sleep_time,
                         1s / (double)limit_fps - std::chrono::duration<double>(cpu_time));
        }

        if (in_flight_data.cpu_sleep_time > 0ms) {
            const auto last_cpu_sleep_time = in_flight_data.cpu_sleep_time;
            in_flight_data.cpu_sleep_time =
                std::min(in_flight_data.cpu_sleep_time,
                         std::chrono::duration<double>(last_cpu_sleep_time * 1.05 + 1ms));
            std::this_thread::sleep_for(in_flight_data.cpu_sleep_time);
        }

        const std::shared_ptr<CachingCommandPool>& cmd_cache = in_flight_data.command_buffer_cache;
        cmd_cache->reset();

        // Compute time stuff
        assert(time_overwrite < 3);
        const std::chrono::nanoseconds last_elapsed_ns = duration_elapsed;
        if (time_overwrite == 1) {
            const auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(time_delta_overwrite_ms / 1000.));
            duration_elapsed += delta;
            duration_elapsed_since_connect += delta;
            time_delta_overwrite_ms = 0;
        } else if (time_overwrite == 2) {
            const auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(time_delta_overwrite_ms / 1000.));
            duration_elapsed += delta;
            duration_elapsed_since_connect += delta;
        } else {
            const auto now = std::chrono::high_resolution_clock::now();
            duration_elapsed = now - time_reference;
            duration_elapsed_since_connect = now - time_connect_reference;
        }
        time_delta = duration_elapsed - last_elapsed_ns;

        const ProfilerHandle& profiler = prepare_profiler_for_run(in_flight_data);
        const auto run_start = std::chrono::high_resolution_clock::now();

        // CONNECT and PREPROCESS
        do {
            // While connection nodes can signalize that they need to reconnect
            while (needs_reconnect) {
                connect();
            }

            graph_run.begin_run(cmd_cache, run_iteration, total_iteration,
                                run_iteration % ITERATIONS_IN_FLIGHT, time_delta, duration_elapsed,
                                duration_elapsed_since_connect);

            // While preprocessing nodes can signalize that they need to reconnect as well
            {
                MERIAN_PROFILE_SCOPE(profiler, "Preprocess nodes");
                for (auto& node : flat_topology) {
                    NodeData& data = node_data.at(node);
                    MERIAN_PROFILE_SCOPE(profiler, fmt::format("{} ({})", data.identifier,
                                                               registry.node_name(node)));
                    const uint32_t set_idx = data.set_index(run_iteration);
                    Node::NodeStatusFlags flags =
                        node->pre_process(graph_run, data.resource_maps[set_idx]);
                    if ((flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                        SPDLOG_DEBUG("node {} requested reconnect in pre_process", data.identifier);
                        request_reconnect();
                    }
                    if ((flags & Node::NodeStatusFlagBits::RESET_IN_FLIGHT_DATA) != 0u) {
                        in_flight_data.in_flight_data[node].reset();
                    }
                }
            }
        } while (needs_reconnect);

        // RUN
        {
            MERIAN_PROFILE_SCOPE(profiler, "on_run_starting");
            on_run_starting(graph_run);
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, graph_run.get_cmd(), "Run nodes");
            for (auto& node : flat_topology) {
                NodeData& data = node_data.at(node);
                if (debug_utils)
                    debug_utils->cmd_begin_label(*graph_run.get_cmd(), registry.node_name(node));

                try {
                    run_node(graph_run, node, data, profiler);
                } catch (const graph_errors::node_error& e) {
                    data.run_errors.emplace_back(fmt::format("node error: {}", e.what()));
                } catch (const ShaderCompiler::compilation_failed& e) {
                    data.run_errors.emplace_back(fmt::format("compilation failed: {}", e.what()));
                }
                if (!data.run_errors.empty()) {
                    SPDLOG_ERROR("executing node '{}' failed:\n - {}", data.identifier,
                                 fmt::join(data.run_errors, "\n   - "));

                    request_reconnect();
                    SPDLOG_ERROR("emergency reconnect.");
                }

                if (debug_utils)
                    debug_utils->cmd_end_label(*graph_run.get_cmd());
            }
        }

        // FINISH RUN: submit

        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, graph_run.get_cmd(), "on_pre_submit");
            on_pre_submit(graph_run);
        }

        {

            MERIAN_PROFILE_SCOPE(profiler, "end run");
            graph_run.end_run(ring_fences.reset());
        }
        {
            MERIAN_PROFILE_SCOPE(profiler, "on_post_submit");
            on_post_submit();
        }

        external_wait_time = 0.9 * external_wait_time + 0.1 * graph_run.external_wait_time;
        needs_reconnect |= graph_run.needs_reconnect;
        ++run_iteration;
        ++total_iteration;
        run_in_progress = false;

        {
            MERIAN_PROFILE_SCOPE(profiler, "on_run_finished_tasks");
            for (const auto& task : on_run_finished_tasks)
                task();
            on_run_finished_tasks.clear();
        }

        cpu_time = std::chrono::high_resolution_clock::now() - run_start;
    }

    // waits until all in-flight iterations have finished
    void wait() {
        SPDLOG_DEBUG("wait until all in-flight iterations have finished");
        ring_fences.wait_all();
        cpu_queue->wait_idle();
    }

    // removes all nodes and connections from the graph.
    void reset() {
        wait();

        node_data.clear();
        node_for_identifier.clear();
        for (uint32_t i = 0; i < ITERATIONS_IN_FLIGHT; i++) {
            InFlightData& in_flight_data = ring_fences.get(i).user_data;
            in_flight_data.in_flight_data.clear();
        }

        needs_reconnect = true;
    }

    // Ensures at reconnect at the next run
    void request_reconnect() {
        needs_reconnect = true;
    }

    bool get_needs_reconnect() {
        return needs_reconnect;
    }

    auto identifiers() {
        return std::as_const(node_for_identifier) | std::ranges::views::keys;
    }

    // --- Events ---

    void send_event(const std::string& event_name,
                    const GraphEvent::Data& data = {},
                    const bool notify_all = true) {
        send_event(GraphEvent::Info{nullptr, "", "user", event_name}, data, notify_all);
    }

    void register_event_listener(const std::string& event_pattern,
                                 const GraphEvent::Listener& event_listener) {
        user_event_pattern_listener.push_back(std::make_pair(event_pattern, event_listener));
    }

    // --- Properties / Graph (de)serialize ---

    void properties(Properties& props) {
        needs_reconnect |= props.config_bool("Rebuild");
        props.st_no_space();
        props.output_text("Run iteration: {}", run_iteration);
        if (props.is_ui() && props.config_text("send event", props_send_event, true) &&
            !props_send_event.empty()) {
            send_event(props_send_event);
            props_send_event.clear();
        }
        if (props.st_begin_child("graph_properties", "Graph Properties",
                                 Properties::ChildFlagBits::FRAMED)) {
            props.output_text("Run iteration: {}", run_iteration);
            props.output_text("Run Elapsed: {:%H:%M:%S}s", duration_elapsed_since_connect);
            props.output_text("Total iterations: {}", total_iteration);
            props.output_text("Total Elapsed: {:%H:%M:%S}s", duration_elapsed);
            props.output_text("Time delta: {:04f}ms", to_milliseconds(time_delta));
            props.output_text("GPU wait: {:04f}ms", to_milliseconds(gpu_wait_time));
            props.output_text("External wait: {:04f}ms", to_milliseconds(external_wait_time));

            props.st_separate();
            if (props.config_options("time overwrite", time_overwrite, {"None", "Time", "Delta"},
                                     Properties::OptionsStyle::COMBO)) {
                if (time_overwrite == 0) {
                    // move reference to prevent jump
                    const auto now = std::chrono::high_resolution_clock::now();
                    time_reference = now - duration_elapsed;
                    time_connect_reference = now - duration_elapsed_since_connect;
                }
            }
            if (time_overwrite == 1) {
                float time_s = to_seconds(duration_elapsed);
                props.config_float("time (s)", time_s, "", 0.1);
                float delta_s = time_s - to_seconds(duration_elapsed);
                props.config_float("offset (s)", delta_s, "", 0.01);
                time_delta_overwrite_ms += delta_s * 1000.;
            } else if (time_overwrite == 2) {
                props.config_float("delta (ms)", time_delta_overwrite_ms, "", 0.001);
                float fps = 1000. / time_delta_overwrite_ms;
                props.config_float("fps", fps, "", 0.01);
                time_delta_overwrite_ms = 1000 / fps;
            }

            props.st_separate();
            if (props.config_bool("fps limiter", limit_fps) && limit_fps != 0) {
                limit_fps = 60;
            }
            if (limit_fps != 0) {
                if (props.config_int("fps limit", limit_fps, "")) {
                    limit_fps = std::max(1, limit_fps);
                }
            }
            props.config_bool(
                "low latency", low_latency_mode,
                "Experimental: Delays CPU processing to recude input latency in GPU bound "
                "applications. Might reduce framerate.");
            if (low_latency_mode || limit_fps > 0) {
                const InFlightData& in_flight_data = ring_fences.get().user_data;
                props.output_text("CPU sleep time: {:04f}ms",
                                  to_milliseconds(in_flight_data.cpu_sleep_time));
            }

            props.st_separate();
            props.config_bool("flush thread pool", flush_thread_pool_at_run_start,
                              "If enabled, the tasks queue of the thread pool is flushed when a "
                              "run starts. HIGHLY RECOOMMENDED as it limits memory allocations and "
                              "prevents the queue to fill up indefinitely.");
            props.output_text("tasks in queue: {}", thread_pool->queue_size());

            props.st_end_child();
        }

        if (props.is_ui() &&
            props.st_begin_child("edit", "Edit Graph", Properties::ChildFlagBits::FRAMED)) {
            props.st_separate("Add Node");
            props.config_options("new type", new_node_selected, registry.node_names(),
                                 Properties::OptionsStyle::COMBO);
            if (props.config_text("new identifier", new_node_identifier, true,
                                  "Set an optional name for the node and press enter.") ||
                props.config_bool("Add Node")) {
                std::optional<std::string> optional_identifier;
                if (!new_node_identifier.empty() && new_node_identifier[0] != 0) {
                    optional_identifier = new_node_identifier;
                }
                add_node(registry.node_names()[new_node_selected], optional_identifier);
            }
            props.output_text(
                "{}: {}", registry.node_names()[new_node_selected],
                registry.node_info(registry.node_names()[new_node_selected]).description);

            const std::vector<std::string> node_ids(identifiers().begin(), identifiers().end());
            props.st_separate("Add Connection");
            bool autodetect_dst_input = false;
            if (props.config_options("connection src", add_connection_selected_src, node_ids,
                                     Properties::OptionsStyle::COMBO)) {
                add_connection_selected_src_output = 0;
                autodetect_dst_input = true;
            }
            std::vector<std::string> src_outputs;
            for (const auto& [output_name, output] :
                 node_data.at(node_for_identifier.at(node_ids[add_connection_selected_src]))
                     .output_connector_for_name) {
                src_outputs.emplace_back(output_name);
                std::sort(src_outputs.begin(), src_outputs.end());
            }
            autodetect_dst_input |=
                props.config_options("connection src output", add_connection_selected_src_output,
                                     src_outputs, Properties::OptionsStyle::COMBO);
            if (props.config_options("connection dst", add_connection_selected_dst, node_ids,
                                     Properties::OptionsStyle::COMBO)) {
                add_connection_selected_dst_input = 0;
                autodetect_dst_input |= true;
            }
            NodeData& dst_data =
                node_data.at(node_for_identifier.at(node_ids[add_connection_selected_dst]));
            std::vector<std::string> dst_inputs;
            dst_inputs.reserve(dst_data.input_connector_for_name.size());
            for (const auto& [input_name, input] : dst_data.input_connector_for_name) {
                dst_inputs.emplace_back(input_name);
            }
            std::sort(dst_inputs.begin(), dst_inputs.end());
            if (autodetect_dst_input &&
                add_connection_selected_src_output < (int)src_outputs.size()) {
                // maybe there is a input that is named exactly like the output
                for (int i = 0; i < static_cast<int>(dst_inputs.size()); i++) {
                    if (dst_inputs[i] == src_outputs[add_connection_selected_src_output]) {
                        add_connection_selected_dst_input = i;
                    }
                }
            }
            props.config_options("connection dst input", add_connection_selected_dst_input,
                                 dst_inputs, Properties::OptionsStyle::COMBO);
            const bool valid_connection =
                add_connection_selected_src_output < (int)src_outputs.size() &&
                add_connection_selected_dst_input < (int)dst_inputs.size();
            if (valid_connection) {
                if (props.config_bool("Add Connection")) {
                    add_connection(node_ids[add_connection_selected_src],
                                   node_ids[add_connection_selected_dst],
                                   src_outputs[add_connection_selected_src_output],
                                   dst_inputs[add_connection_selected_dst_input]);
                }

                const auto it = dst_data.desired_incoming_connections.find(
                    dst_inputs[add_connection_selected_dst_input]);
                if (it != dst_data.desired_incoming_connections.end()) {
                    props.st_no_space();
                    props.output_text("Warning: Input already connected with {}, {} ({})",
                                      it->second.second, node_data.at(it->second.first).identifier,
                                      registry.node_name(it->second.first));
                }
            }
            props.st_separate("Remove Node");
            props.config_options("remove identifier", remove_node_selected, node_ids,
                                 Properties::OptionsStyle::COMBO);
            if (props.config_bool("Remove Node")) {
                remove_node(node_ids[remove_node_selected]);
            }

            props.st_end_child();
        }

        if (props.st_begin_child("profiler", "Profiler", Properties::ChildFlagBits::FRAMED)) {
#ifdef MERIAN_PROFILER_ENABLE
            props.config_bool("profiling", profiler_enable);
#else
            profiler_enable = false;
            props.output_text("Profiler disabled at compile-time!\n\n Enable with 'meson configure "
                              "<builddir> -Dmerian:performance_profiling=true'.");
#endif

            if (profiler_enable) {
                props.st_no_space();
                props.config_uint("report intervall", profiler_report_intervall_ms,
                                  "Set the time period for the profiler to update in ms. Meaning, "
                                  "averages and deviations are calculated over this this period.");

                if (last_run_report &&
                    props.st_begin_child("run", "Graph Run",
                                         Properties::ChildFlagBits::DEFAULT_OPEN)) {
                    if (!last_run_report.cpu_report.empty()) {
                        props.st_separate("CPU");
                        const float* cpu_samples = &cpu_time_history[time_history_current + 1];
                        if (cpu_auto) {
                            cpu_max = *std::max_element(cpu_samples,
                                                        cpu_samples + cpu_time_history.size() - 1);
                        }

                        props.output_plot_line("", cpu_samples, cpu_time_history.size() - 1, 0,
                                               cpu_max);
                        cpu_auto &= !props.config_float("cpu max ms", cpu_max, 0, 1000);
                        props.st_no_space();
                        props.config_bool("cpu auto", cpu_auto);
                        Profiler::get_cpu_report_as_config(props, last_run_report);
                    }

                    if (!last_run_report.gpu_report.empty()) {
                        props.st_separate("GPU");
                        const float* gpu_samples = &gpu_time_history[time_history_current + 1];
                        if (gpu_auto) {
                            gpu_max = *std::max_element(gpu_samples,
                                                        gpu_samples + gpu_time_history.size() - 1);
                        }

                        props.output_plot_line("", gpu_samples, gpu_time_history.size() - 1, 0,
                                               gpu_max);
                        gpu_auto &= !props.config_float("gpu max ms", gpu_max, 0, 1000);
                        props.st_no_space();
                        props.config_bool("gpu auto", gpu_auto);
                        Profiler::get_gpu_report_as_config(props, last_run_report);
                    }
                    props.st_end_child();
                }
                if (last_build_report && props.st_begin_child("build", "Last Graph Build")) {
                    Profiler::get_report_as_config(props, last_build_report);
                    props.st_end_child();
                }
            }
            props.st_end_child();
        }

        bool loading = false;
        if (props.st_begin_child("nodes", "Nodes",
                                 Properties::ChildFlagBits::DEFAULT_OPEN |
                                     Properties::ChildFlagBits::FRAMED)) {
            std::vector<std::string> nodes(identifiers().begin(), identifiers().end());

            if (nodes.empty() && !props.is_ui()) {
                nodes = props.st_list_children();

                if (!nodes.empty()) {
                    // go into "loading" mode
                    SPDLOG_INFO("Reconstructing graph from properties.");
                    loading = true;
                    reset(); // never know...
                }
            }

            for (const auto& identifier : nodes) {

                std::string node_label;
                if (!loading) {
                    // otherwise the node data does not exist!
                    const NodeHandle& node = node_for_identifier.at(identifier);
                    const auto& data = node_data.at(node);
                    std::string state = "OK";
                    if (data.disable) {
                        state = "DISABLED";
                    } else if (!data.errors.empty()) {
                        state = "ERROR";
                    }

                    node_label = fmt::format("[{}] {} ({})", state, data.identifier,
                                             registry.node_name(node));
                }

                if (props.st_begin_child(identifier, node_label)) {
                    NodeHandle node;
                    std::string type;

                    // Create Node
                    if (!loading) {
                        node = node_for_identifier.at(identifier);
                        type = registry.node_name(node);
                    }
                    props.serialize_string("type", type);
                    if (loading) {
                        node = node_for_identifier.at(add_node(type, identifier));
                    }
                    NodeData& data = node_data.at(node);

                    if (props.config_bool("disable", data.disable))
                        request_reconnect();
                    props.st_no_space();
                    if (props.config_bool("Remove")) {
                        remove_node(identifier);
                    }

                    if (!data.errors.empty()) {
                        props.output_text(
                            fmt::format("Errors:\n  - {}", fmt::join(data.errors, "\n   - ")));
                    }
                    props.st_separate();
                    if (props.st_begin_child("properties", "Properties",
                                             Properties::ChildFlagBits::DEFAULT_OPEN)) {
                        const Node::NodeStatusFlags flags = node->properties(props);
                        if ((flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                            SPDLOG_DEBUG("node {} requested reconnect", data.identifier);
                            request_reconnect();
                        }
                        props.st_end_child();
                    }
                    if (props.st_begin_child("stats", "Statistics")) {
                        props.output_text(fmt::format("{}", data.statistics));
                        props.st_end_child();
                    };
                    io_props_for_node(props, node, data);
                    props.st_end_child();
                }
            }
            props.st_end_child();
        }

        if (!props.is_ui()) {
            nlohmann::json connections;
            if (!loading) {
                for (const auto& identifier : identifiers()) {
                    const NodeHandle& node = node_for_identifier.at(identifier);
                    const auto& data = node_data.at(node);
                    for (const OutgoingNodeConnection& con : data.desired_outgoing_connections) {
                        nlohmann::json j_con;
                        j_con["src"] = data.identifier;
                        j_con["dst"] = node_data.at(con.dst).identifier;
                        j_con["src_output"] = con.src_output;
                        j_con["dst_input"] = con.dst_input;

                        connections.push_back(j_con);
                    }
                }
            }
            std::sort(connections.begin(), connections.end());
            props.serialize_json("connections", connections);
            if (loading) {
                for (auto& j_con : connections) {
                    add_connection(j_con["src"], j_con["dst"], j_con["src_output"],
                                   j_con["dst_input"]);
                }
            }
        }
    }

  private:
    /*--- Graph Edit --- */

    const std::string& add_node(const std::shared_ptr<Node>& node,
                                const std::optional<std::string>& identifier = std::nullopt) {
        if (node_data.contains(node)) {
            throw std::invalid_argument{fmt::format("graph already contains this node as '{}'",
                                                    node_data.at(node).identifier)};
        }

        std::string node_identifier;
        if (identifier) {
            if (identifier->empty()) {
                throw std::invalid_argument{"node identifier cannot be empty"};
            }
            if (node_for_identifier.contains(identifier.value())) {
                throw std::invalid_argument{fmt::format(
                    "graph already contains a node with identifier '{}'", identifier.value())};
            }
            if (*identifier == "user") {
                throw std::invalid_argument{"the identifier 'user' is reserved"};
            }
            if (*identifier == "graph") {
                throw std::invalid_argument{"the identifier 'graph' is reserved"};
            }
            node_identifier = identifier.value();
        } else {
            uint32_t i = 0;
            do {
                node_identifier = fmt::format("{} {}", registry.node_name(node), i++);
            } while (node_for_identifier.contains(node_identifier));
        }

        node_for_identifier[node_identifier] = node;
        auto [it, inserted] = node_data.try_emplace(node, node_identifier);
        assert(inserted);

        needs_reconnect = true;
        SPDLOG_DEBUG("added node {} ({})", node_identifier, registry.node_name(node));

        return it->second.identifier;
    }

    void add_connection(const NodeHandle& src,
                        const NodeHandle& dst,
                        const std::string& src_output,
                        const std::string& dst_input) {
        if (!node_data.contains(src) || !node_data.contains(dst)) {
            throw std::invalid_argument{"graph does not contain the source or destination node"};
        }

        NodeData& src_data = node_data.at(src);
        NodeData& dst_data = node_data.at(dst);

        if (dst_data.desired_incoming_connections.contains(dst_input)) {
            const auto& [old_src, old_src_output] =
                dst_data.desired_incoming_connections.at(dst_input);
            [[maybe_unused]] const NodeData& old_src_data = node_data.at(old_src);
            SPDLOG_DEBUG("remove conflicting connection {}, {} ({}) -> {}, {} ({})", old_src_output,
                         old_src_data.identifier, registry.node_name(old_src), dst_input,
                         dst_data.identifier, registry.node_name(dst));
            remove_connection(old_src, dst, dst_input);
        }

        {
            // outgoing
            [[maybe_unused]] const auto [it, inserted] =
                src_data.desired_outgoing_connections.emplace(dst, src_output, dst_input);
            assert(inserted);
        }

        {
            // incoming
            [[maybe_unused]] const auto [it, inserted] =
                dst_data.desired_incoming_connections.try_emplace(dst_input, src, src_output);
            assert(inserted);
        }

        needs_reconnect = true;
        SPDLOG_DEBUG("added connection {}, {} ({}) -> {}, {} ({})", src_output, src_data.identifier,
                     registry.node_name(src), dst_input, dst_data.identifier,
                     registry.node_name(dst));
    }

    bool
    remove_connection(const NodeHandle src, const NodeHandle dst, const std::string dst_input) {
        // Developer note: Pass by reference is not used because this might be called with
        // references to iterator objects of the sets/maps we edit.

        if (!node_data.contains(src) || !node_data.contains(dst)) {
            throw std::invalid_argument{"graph does not contain the source or destination node"};
        }
        NodeData& src_data = node_data.at(src);
        NodeData& dst_data = node_data.at(dst);

        const auto it = dst_data.desired_incoming_connections.find(dst_input);
        if (it == dst_data.desired_incoming_connections.end()) {
            SPDLOG_WARN("connection {} ({}) -> {}, {} ({}) does not exist and cannot be removed.",
                        src_data.identifier, registry.node_name(src), dst_input,
                        dst_data.identifier, registry.node_name(dst));
            return false;
        }

        const std::string src_output = it->second.second;
        dst_data.desired_incoming_connections.erase(it);

        const auto out_it =
            src_data.desired_outgoing_connections.find({dst, src_output, dst_input});
        // else we did not add the connection properly
        assert(out_it != src_data.desired_outgoing_connections.end());
        src_data.desired_outgoing_connections.erase(out_it);
        SPDLOG_DEBUG("removed connection {}, {} ({}) -> {}, {} ({})", src_output,
                     src_data.identifier, registry.node_name(src), dst_input, dst_data.identifier,
                     registry.node_name(dst));

        needs_reconnect = true;
        return true;

        // Note: Since the connections are not needed in a graph run we do not need to wait until
        // the end of a run to remove the conenction.
    }

    /*--- Properties ---*/
    void io_props_for_node(Properties& config, NodeHandle& node, NodeData& data) {
        if (data.descriptor_set_layout &&
            config.st_begin_child("desc_set_layout", "Descriptor Set Layout")) {
            config.output_text(fmt::format("{}", data.descriptor_set_layout));
            config.st_end_child();
        }
        if (!data.output_connections.empty() && config.st_begin_child("outputs", "Outputs")) {
            for (auto& [output, per_output_info] : data.output_connections) {
                if (config.st_begin_child(output->name, output->name)) {
                    std::vector<std::string> receivers;
                    receivers.reserve(per_output_info.inputs.size());
                    for (auto& [node, input] : per_output_info.inputs) {
                        receivers.emplace_back(fmt::format("({}, {} ({}))", input->name,
                                                           node_data.at(node).identifier,
                                                           registry.node_name(node)));
                    }

                    std::string current_resource_index = "none";
                    if (!per_output_info.precomputed_resources.empty()) {
                        const uint32_t set_idx = data.set_index(run_iteration);
                        current_resource_index = fmt::format(
                            "{:02}", std::get<1>(per_output_info.precomputed_resources[set_idx]));
                    }

                    config.output_text(fmt::format(
                        "Descriptor set binding: {}\n# Resources: {:02}\nResource index: "
                        "{}\nSending to: [{}]",
                        per_output_info.descriptor_set_binding == NodeData::NO_DESCRIPTOR_BINDING
                            ? "none"
                            : std::to_string(per_output_info.descriptor_set_binding),
                        per_output_info.resources.size(), current_resource_index,
                        fmt::join(receivers, ", ")));

                    config.st_separate("Connector Properties");
                    output->properties(config);
                    config.st_separate("Resource Properties");
                    for (uint32_t i = 0; i < per_output_info.resources.size(); i++) {
                        if (config.st_begin_child(fmt::format("resource_{}", i),
                                                  fmt::format("Resource {:02}", i))) {
                            per_output_info.resources[i].resource->properties(config);
                            config.st_end_child();
                        }
                    }

                    config.st_end_child();
                }
            }
            config.st_end_child();
        }
        if (!data.input_connectors.empty() && config.st_begin_child("inputs", "Inputs")) {
            for (const auto& input : data.input_connectors) {
                if (config.st_begin_child(input->name, input->name)) {
                    config.st_separate("Input Properties");
                    input->properties(config);
                    config.st_separate("Connection");
                    if (data.input_connections.contains(input)) {
                        auto& per_input_info = data.input_connections.at(input);
                        config.output_text(fmt::format(
                            "Descriptor set binding: {}",
                            per_input_info.descriptor_set_binding == NodeData::NO_DESCRIPTOR_BINDING
                                ? "None"
                                : std::to_string(per_input_info.descriptor_set_binding)));
                        if (per_input_info.output) {
                            config.output_text(fmt::format(
                                "Receiving from: {}, {} ({})", per_input_info.output->name,
                                node_data.at(per_input_info.node).identifier,
                                registry.node_name(per_input_info.node)));
                        } else {
                            config.output_text("Optional input not connected.");
                        }
                    } else {
                        config.output_text("Input not connected.");
                    }

                    if (data.desired_incoming_connections.contains(input->name) &&
                        config.config_bool("Remove Connection")) {
                        auto& incoming_conection =
                            data.desired_incoming_connections.at(input->name);
                        remove_connection(incoming_conection.first, node, input->name);
                    }

                    config.st_end_child();
                }
            }
            config.st_end_child();
        }
    }

    // --- Graph run sub-tasks ---

    // Creates the profiler if necessary
    ProfilerHandle prepare_profiler_for_run(InFlightData& in_flight_data) {
        if (!profiler_enable) {
            last_run_report = {};
            return nullptr;
        }

        auto report = run_profiler->set_collect_get_every(in_flight_data.profiler_query_pool,
                                                          profiler_report_intervall_ms);

        if (report) {
            last_run_report = std::move(*report);
            cpu_time_history.set(time_history_current, last_run_report.cpu_total());
            gpu_time_history.set(time_history_current, last_run_report.gpu_total());
            time_history_current++;
        }

        return run_profiler;
    }

    // Calls connector callbacks, checks resource states and records as well as applies descriptor
    // set updates.
    void run_node(GraphRun& run,
                  const NodeHandle& node,
                  NodeData& data,
                  [[maybe_unused]] const ProfilerHandle& profiler) {
        const uint32_t set_idx = data.set_index(run_iteration);

        MERIAN_PROFILE_SCOPE_GPU(profiler, run.get_cmd(),
                                 fmt::format("{} ({})", data.identifier, registry.node_name(node)));

        std::vector<vk::ImageMemoryBarrier2> image_barriers;
        std::vector<vk::BufferMemoryBarrier2> buffer_barriers;

        {
            // Call connector callbacks (pre_process) and record descriptor set updates
            for (auto& [input, per_input_info] : data.input_connections) {
                if (!per_input_info.node) {
                    // optional input not connected
                    continue;
                }

                auto& [resource, resource_index] = per_input_info.precomputed_resources[set_idx];
                const Connector::ConnectorStatusFlags flags = input->on_pre_process(
                    run, run.get_cmd(), resource, node, image_barriers, buffer_barriers);
                if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) != 0u) {
                    NodeData& src_data = node_data.at(per_input_info.node);
                    record_descriptor_updates(src_data, per_input_info.output,
                                              src_data.output_connections[per_input_info.output],
                                              resource_index);
                }
                if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                    SPDLOG_DEBUG("input connector {} at node {} requested reconnect.", input->name,
                                 data.identifier);
                    request_reconnect();
                }
            }
            for (auto& [output, per_output_info] : data.output_connections) {
                auto& [resource, resource_index] = per_output_info.precomputed_resources[set_idx];
                const Connector::ConnectorStatusFlags flags = output->on_pre_process(
                    run, run.get_cmd(), resource, node, image_barriers, buffer_barriers);
                if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) != 0u) {
                    record_descriptor_updates(data, output, per_output_info, resource_index);
                }
                if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                    SPDLOG_DEBUG("output connector {} at node {} requested reconnect.",
                                 output->name, data.identifier);
                    request_reconnect();
                }
            }

            if (!image_barriers.empty() || !buffer_barriers.empty()) {
                vk::DependencyInfoKHR dep_info{{}, {}, buffer_barriers, image_barriers};
                run.get_cmd()->barrier({}, buffer_barriers, image_barriers);
                image_barriers.clear();
                buffer_barriers.clear();
            }
        }

        auto& descriptor_set = data.descriptor_sets[set_idx];
        {
            // apply descriptor set updates
            data.statistics.last_descriptor_set_updates = descriptor_set->update_count();
            if (descriptor_set->has_updates()) {
                SPDLOG_TRACE("applying {} descriptor set updates for node {}, set {}",
                             descriptor_set.update->count(), data.name, set_idx);
                descriptor_set->update();
            }
        }

        {
            node->process(run, descriptor_set, data.resource_maps[set_idx]);
#ifndef NDEBUG
            if (run.needs_reconnect && !get_needs_reconnect()) {
                SPDLOG_DEBUG("node {} requested reconnect in process", data.identifier);
                request_reconnect();
            }
#endif
        }

        {
            // Call connector callbacks (post_process) and record descriptor set updates
            for (auto& [input, per_input_info] : data.input_connections) {
                if (!per_input_info.node) {
                    // optional input not connected
                    continue;
                }

                auto& [resource, resource_index] = per_input_info.precomputed_resources[set_idx];
                const Connector::ConnectorStatusFlags flags = input->on_post_process(
                    run, run.get_cmd(), resource, node, image_barriers, buffer_barriers);
                if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) != 0u) {
                    NodeData& src_data = node_data.at(per_input_info.node);
                    record_descriptor_updates(src_data, per_input_info.output,
                                              src_data.output_connections[per_input_info.output],
                                              resource_index);
                }
                if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                    SPDLOG_DEBUG("input connector {} at node {} requested reconnect.", input->name,
                                 data.identifier);
                    request_reconnect();
                }
            }
            for (auto& [output, per_output_info] : data.output_connections) {
                auto& [resource, resource_index] = per_output_info.precomputed_resources[set_idx];
                const Connector::ConnectorStatusFlags flags = output->on_post_process(
                    run, run.get_cmd(), resource, node, image_barriers, buffer_barriers);
                if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) != 0u) {
                    record_descriptor_updates(data, output, per_output_info, resource_index);
                }
                if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                    SPDLOG_DEBUG("output connector {} at node {} requested reconnect.",
                                 output->name, data.identifier);
                    request_reconnect();
                }
            }

            if (!image_barriers.empty() || !buffer_barriers.empty()) {
                run.get_cmd()->barrier({}, buffer_barriers, image_barriers);
            }
        }
    }

    void record_descriptor_updates(NodeData& src_data,
                                   const OutputConnectorHandle& src_output,
                                   NodeData::PerOutputInfo& per_output_info,
                                   const uint32_t resource_index) {
        NodeData::PerResourceInfo& resource_info = per_output_info.resources[resource_index];

        if (per_output_info.descriptor_set_binding != NodeData::NO_DESCRIPTOR_BINDING)
            for (auto& set_idx : resource_info.set_indices) {
                src_output->get_descriptor_update(
                    per_output_info.descriptor_set_binding, resource_info.resource,
                    src_data.descriptor_sets[set_idx], resource_allocator);
            }

        for (auto& [dst_node, dst_input, set_idx] : resource_info.other_set_indices) {
            NodeData& dst_data = node_data.at(dst_node);
            NodeData::PerInputInfo& per_input_info = dst_data.input_connections[dst_input];
            if (per_input_info.descriptor_set_binding != NodeData::NO_DESCRIPTOR_BINDING)
                dst_input->get_descriptor_update(
                    per_input_info.descriptor_set_binding, resource_info.resource,
                    dst_data.descriptor_sets[set_idx], resource_allocator);
        }
    }

    // --- Graph connect sub-tasks ---

    // Removes all connections, frees graph resources and resets the precomputed topology.
    // Only keeps desired connections.
    void reset_connections() {
        SPDLOG_DEBUG("reset connections");

        flat_topology.clear();
        maybe_connected_inputs.clear();
        for (auto& [node, data] : node_data) {
            data.reset();
        }
        event_listeners.clear();
    }

    // Calls the describe_inputs() methods of the nodes and caches the result in
    // - node_data[].input_connectors
    // - node_data[].input_connector_for_name
    // - maybe_connected_inputs
    //
    // For all desired_connections: Ensures that the inputs exists (outputs are NOT checked)
    // and that the connections to all inputs are unique.
    [[nodiscard]]
    bool cache_node_input_connectors() {
        for (auto& [node, data] : node_data) {
            // Cache input connectors in node_data and check that there are no name conflicts.
            try {
                data.input_connectors = node->describe_inputs();
            } catch (const graph_errors::node_error& e) {
                data.errors.emplace_back(fmt::format("node error: {}", e.what()));
            } catch (const ShaderCompiler::compilation_failed& e) {
                data.errors.emplace_back(fmt::format("compilation failed: {}", e.what()));
            }
            for (const InputConnectorHandle& input : data.input_connectors) {
                if (data.input_connector_for_name.contains(input->name)) {
                    throw graph_errors::connector_error{
                        fmt::format("node {} contains two input connectors with the same name {}",
                                    registry.node_name(node), input->name)};
                }
                data.input_connector_for_name[input->name] = input;
            }
        }

        // Store connectors that might be connected (there may still be an invalid connection...)
        for (auto& [node, data] : node_data) {
            for (const auto& connection : data.desired_outgoing_connections) {
                NodeData& dst_data = node_data.at(connection.dst);
                if (!dst_data.errors.empty()) {
                    SPDLOG_WARN("node {} has errors and connection {}, {} ({}) -> {}, {} ({}) "
                                "cannot be validated.",
                                dst_data.identifier, connection.src_output, data.identifier,
                                registry.node_name(node), connection.dst_input, dst_data.identifier,
                                registry.node_name(connection.dst));
                    continue;
                }
                if (!dst_data.input_connector_for_name.contains(connection.dst_input)) {
                    SPDLOG_ERROR("node {} ({}) does not have an input {}. Connection is removed.",
                                 dst_data.identifier, registry.node_name(connection.dst),
                                 connection.dst_input);
                    remove_connection(node, connection.dst, connection.dst_input);
                    return false;
                }
                if (connection.dst == node &&
                    dst_data.input_connector_for_name.at(connection.dst_input)->delay == 0) {
                    // eliminate self loops
                    SPDLOG_ERROR("undelayed (edges with delay = 0) selfloop {} -> {} detected on "
                                 "node {}! Removing connection.",
                                 data.identifier, connection.src_output, connection.dst_input);
                    remove_connection(node, connection.dst, connection.dst_input);
                    return false;
                }

                const InputConnectorHandle& dst_input =
                    dst_data.input_connector_for_name[connection.dst_input];
                [[maybe_unused]] const auto [it, inserted] =
                    maybe_connected_inputs.try_emplace(dst_input, node);

                assert(inserted); // uniqueness should be made sure in add_connection!
            }
        }

        return true;
    }

    // Only for a "satisfied node". Means, all inputs are connected, or delayed or optional and will
    // not be connected.
    void cache_node_output_connectors(const NodeHandle& node, NodeData& data) {
        try {
            data.output_connectors = node->describe_outputs(NodeIOLayout(
                [&](const InputConnectorHandle& input) {
#ifndef NDEBUG
                    if (input->delay > 0) {
                        throw std::runtime_error{fmt::format(
                            "Node {} tried to access an output connector that is connected "
                            "through a delayed input {} (which is not allowed here but only in "
                            "on_connected).",
                            registry.node_name(node), input->name)};
                    }
                    if (std::find(data.input_connectors.begin(), data.input_connectors.end(),
                                  input) == data.input_connectors.end()) {
                        throw std::runtime_error{
                            fmt::format("Node {} tried to get an output connector for an input {} "
                                        "which was not returned in describe_inputs (which is not "
                                        "how this works).",
                                        registry.node_name(node), input->name)};
                    }
#endif
                    // for optional inputs we inserted a input connection with nullptr in
                    // search_satisfied_nodes, no problem here.
                    return data.input_connections.at(input).output;
                },
                [&](const std::string& event_pattern, const GraphEvent::Listener& listener) {
                    register_event_listener_for_connect(event_pattern, listener);
                }));
        } catch (const graph_errors::node_error& e) {
            data.errors.emplace_back(fmt::format("node error: {}", e.what()));
        } catch (const ShaderCompiler::compilation_failed& e) {
            data.errors.emplace_back(fmt::format("compilation failed: {}", e.what()));
        }

        for (const auto& output : data.output_connectors) {
#ifndef NDEBUG
            if (!output) {
                SPDLOG_CRITICAL("node {} ({}) returned nullptr in describe_outputs",
                                data.identifier, registry.node_name(node));
                assert(output && "node returned nullptr in describe_outputs");
            }
#endif
            if (data.output_connector_for_name.contains(output->name)) {
                throw graph_errors::connector_error{
                    fmt::format("node {} contains two output connectors with the same name {}",
                                registry.node_name(node), output->name)};
            }
            data.output_connector_for_name.try_emplace(output->name, output);
            data.output_connections.try_emplace(output);
        }
    }

    [[nodiscard]]
    bool connect_node(const NodeHandle& node,
                      NodeData& data,
                      const std::unordered_set<NodeHandle>& visited) {
        assert(visited.contains(node) && "necessary for self loop check");
        assert(data.errors.empty() && !data.disable);

        for (const OutgoingNodeConnection& connection : data.desired_outgoing_connections) {
            // since the node is not disabled and not in error state we know the outputs are valid.
            if (!data.output_connector_for_name.contains(connection.src_output)) {
                SPDLOG_ERROR("node {} ({}) does not have an output {}. Removing connection.",
                             data.identifier, registry.node_name(node), connection.src_output);
                remove_connection(node, connection.dst, connection.dst_input);
                return false;
            }
            const OutputConnectorHandle src_output =
                data.output_connector_for_name[connection.src_output];
            NodeData& dst_data = node_data.at(connection.dst);
            if (dst_data.disable) {
                SPDLOG_DEBUG("skipping connection to disabled node {}, {} ({})",
                             connection.dst_input, dst_data.identifier,
                             registry.node_name(connection.dst));
                continue;
            }
            if (!dst_data.errors.empty()) {
                SPDLOG_WARN("skipping connection to erroneous node {}, {} ({})",
                            connection.dst_input, dst_data.identifier,
                            registry.node_name(connection.dst));
                continue;
            }
            if (!dst_data.input_connector_for_name.contains(connection.dst_input)) {
                // since the node is not disabled and not in error state we know the inputs are
                // valid.
                SPDLOG_ERROR("node {} ({}) does not have an input {}. Removing connection.",
                             dst_data.identifier, registry.node_name(connection.dst),
                             connection.dst_input);
                remove_connection(node, connection.dst, connection.dst_input);
                return false;
            }
            const InputConnectorHandle dst_input =
                dst_data.input_connector_for_name[connection.dst_input];

            // made sure in cache_node_input_connectors
            assert(!dst_data.input_connections.contains(dst_input));

            // self loops should be elimited in cache_node_input_connectors.
            if (dst_input->delay == 0 && visited.contains(connection.dst)) {
                // Back-edges with delay > 1 are allowed!
                SPDLOG_ERROR("undelayed (edges with delay = 0) graph is not "
                             "acyclic! {} -> {}. Removing arbitraty edge on the cycle.",
                             data.identifier, node_data.at(connection.dst).identifier);
                remove_connection(node, connection.dst, connection.dst_input);
                return false;
            }

            if (!src_output->supports_delay && dst_input->delay > 0) {
                SPDLOG_ERROR("input connector {} of node {} ({}) was connected to output "
                             "connector {} on node {} ({}) with delay {}, however the output "
                             "connector does not support delay. Removing connection.",
                             dst_input->name, dst_data.identifier,
                             registry.node_name(connection.dst), src_output->name, data.identifier,
                             registry.node_name(node), dst_input->delay);
                remove_connection(node, connection.dst, connection.dst_input);
                return false;
            }

            dst_data.input_connections.try_emplace(dst_input,
                                                   NodeData::PerInputInfo{node, src_output});
            data.output_connections[src_output].inputs.emplace_back(connection.dst, dst_input);
        }

        return true;
    }

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
                                std::priority_queue<NodeHandle>& queue) {
        std::vector<NodeHandle> to_erase;
        // find nodes with all inputs conencted, delayed, or optional and will not be connected
        for (const NodeHandle& node : candidates) {
            NodeData& data = node_data.at(node);

            if (data.disable) {
                SPDLOG_DEBUG("node {} ({}) is disabled, skipping...", data.identifier,
                             registry.node_name(node));
                to_erase.push_back(node);
                continue;
            }
            if (!data.run_errors.empty()) {
                SPDLOG_DEBUG("node {} ({}) has run errors, converting to build errors.");
                move_all(data.errors, data.run_errors);
                data.run_errors.clear();
            }
            if (!data.errors.empty()) {
                SPDLOG_DEBUG("node {} ({}) is erroneous, skipping...", data.identifier,
                             registry.node_name(node));
                to_erase.push_back(node);
                continue;
            }

            bool satisfied = true;
            for (const auto& input : data.input_connectors) {
                // is there a connection to this input possible?
                bool will_not_connect = false;

                if (!maybe_connected_inputs.contains(input)) {
                    will_not_connect = true;
                } else {
                    const NodeHandle& connecting_node = maybe_connected_inputs[input];
                    const NodeData& connecting_node_data = node_data.at(connecting_node);

                    if (connecting_node_data.disable || !connecting_node_data.errors.empty()) {
                        will_not_connect = true;
                    }
                }

                if (will_not_connect) {
                    if (input->delay > 0) {
                        // Special case: We could remove the node here already since it will
                        // never be fully connected. However we might want to know the outputs
                        // of the node for GUI and technically the node is "satisfied" for a
                        // call to describe_outputs.
                        // Note: We cannot set the error here since that would lead to other nodes
                        // not connecting other inputs.
                    } else if (!input->optional) {
                        // This is bad. No node will connect to this input and the input is not
                        // optional...
                        std::string error = make_error_input_not_connected(input, node, data);
                        SPDLOG_WARN(error);
                        data.errors.emplace_back(std::move(error));

                        // We can't even call describe_outputs... Kill the node.
                        to_erase.push_back(node);
                        satisfied = false;
                        break;
                    } else {
                        // We can save this. No node will connect to this input but the input is
                        // optional. Mark the input as optional and unconnected.
                        data.input_connections.try_emplace(
                            input, graph_internal::NodeData::PerInputInfo());
                    }
                } else {
                    // Something will connect to this node, eventually.
                    // We can process this node if the input is either delayed or already connected
                    satisfied &= (data.input_connections.contains(input)) || (input->delay > 0);
                }
            }

            if (satisfied) {
                queue.push(node);
                to_erase.push_back(node);
            }
        }

        for (const NodeHandle& node : to_erase) {
            candidates.erase(node);
        }
    }

    // Attemps to connect the graph from the desired connections.
    // Returns a topological order in which the nodes can be executed, which only includes
    // non-disabled nodes.
    //
    // Returns false if failed and needs reconnect.
    bool connect_nodes() {
        SPDLOG_DEBUG("connecting nodes");

        if (!cache_node_input_connectors()) {
            return false;
        }

        assert(flat_topology.empty());
        flat_topology.reserve(node_data.size());

        // nodes that are active, and were visited.
        std::unordered_set<NodeHandle> visited;
        // nodes that might be active but could not be checked yet.
        std::set<NodeHandle> candidates;

        for (const auto& [node, data] : node_data) {
            candidates.insert(node);
        }

        std::priority_queue<NodeHandle> queue;
        while (!candidates.empty()) {

            search_satisfied_nodes(candidates, queue);

            while (!queue.empty()) {
                const NodeHandle node = queue.top();
                queue.pop();

                visited.insert(node);
                NodeData& data = node_data.at(node);

                {
                    assert(!data.disable && data.errors.empty());
                    SPDLOG_DEBUG("connecting {} ({})", data.identifier, registry.node_name(node));

                    // 1. Get node output connectors and check for name conflicts
                    cache_node_output_connectors(node, data);

                    if (!data.errors.empty()) {
                        // something went wrong earlier (eg. node threw in describe outputs).
                        continue;
                    }

                    // 2. Connect outputs to the inputs of destination nodes (fill in their
                    // input_connections and the current nodes output_connections).
                    if (!connect_node(node, data, visited)) {
                        return false;
                    }

                    flat_topology.emplace_back(node);
                }
            }
        }

        // Now it might be possible that a node later in the topolgy was disabled and thus the
        // backward edge does not exist. Therefore we need to traverse the topology and disable
        // those nodes iteratively. Multiple times since disabled nodes, can have backward edges
        // themselfes...
        {
            std::vector<NodeHandle> filtered_topology;
            filtered_topology.reserve(flat_topology.size());

            for (bool changed = true; changed;) {
                changed = false;
                filtered_topology.clear();

                for (const auto& node : flat_topology) {
                    NodeData& data = node_data.at(node);
                    assert(!data.disable);
                    for (const auto& input : data.input_connectors) {
                        if (!data.input_connections.contains(input)) {
                            if (input->optional) {
                                data.input_connections.try_emplace(input, NodeData::PerInputInfo());
                            } else {
                                // Not connected delayed inputs are filtered here.
                                std::string error =
                                    make_error_input_not_connected(input, node, data);
                                data.errors.emplace_back(error);
                                SPDLOG_WARN(error);
                            }
                        } else {
                            NodeData::PerInputInfo& input_info = data.input_connections[input];
                            if (input_info.node && !node_data.at(input_info.node).errors.empty()) {
                                if (input->optional) {
                                    data.input_connections[input] = NodeData::PerInputInfo();
                                } else {
                                    data.input_connections.erase(input);
                                    std::string error =
                                        make_error_input_not_connected(input, node, data);
                                    SPDLOG_WARN(error);
                                    data.errors.emplace_back(std::move(error));
                                }
                            }
                        }
                    }

                    if (data.errors.empty()) {
                        filtered_topology.emplace_back(node);
                    } else {
                        changed = true;
                    }
                }

                std::swap(filtered_topology, flat_topology);
            };
        }

        // Now clean up this mess. All output connections going to disabled nodes must be
        // eliminated. And finally also call the connector callbacks.
        for (const auto& src_node : flat_topology) {
            NodeData& src_data = node_data.at(src_node);

            for (auto& [src_output, per_output_info] : src_data.output_connections) {
                for (auto it = per_output_info.inputs.begin();
                     it != per_output_info.inputs.end();) {

                    const auto& [dst_node, dst_input] = *it;
                    const auto& dst_data = node_data.at(dst_node);
                    if (!dst_data.errors.empty()) {
                        SPDLOG_TRACE("cleanup output connection to erroneous node: {}, {} ({}) -> "
                                     "{}, {} ({})",
                                     src_output->name, src_data.name, src_node->name,
                                     dst_input->name, dst_data.name, dst_node->name);
                        it = per_output_info.inputs.erase(it);
                    } else {
                        try {
                            src_output->on_connect_input(dst_input);
                            dst_input->on_connect_output(src_output);
                        } catch (const graph_errors::invalid_connection& e) {
                            SPDLOG_ERROR("Removing invalid connection {}, {} ({}) -> {}, {} ({}). "
                                         "Reason: {}",
                                         src_output->name, src_data.identifier,
                                         registry.node_name(src_node), dst_input->name,
                                         dst_data.identifier, registry.node_name(dst_node),
                                         e.what());
                            remove_connection(src_node, dst_node, dst_input->name);
                            return false;
                        }
                        ++it;
                    }
                }
            }
        }

        return true;
    }

    void allocate_resources() {
        for (const auto& node : flat_topology) {
            auto& data = node_data.at(node);
            for (auto& [output, per_output_info] : data.output_connections) {
                uint32_t max_delay = 0;
                for (auto& input : per_output_info.inputs) {
                    max_delay = std::max(max_delay, std::get<1>(input)->delay);
                }

                SPDLOG_DEBUG("creating, connecting and allocating {} resources for output {} on "
                             "node {} ({})",
                             max_delay + 1, output->name, data.identifier,
                             registry.node_name(node));
                for (uint32_t i = 0; i <= max_delay; i++) {
                    const GraphResourceHandle res =
                        output->create_resource(per_output_info.inputs, resource_allocator,
                                                resource_allocator, i, ITERATIONS_IN_FLIGHT);
                    per_output_info.resources.emplace_back(res);
                }
            }
        }
    }

    void prepare_descriptor_sets() {
        for (auto& dst_node : flat_topology) {
            auto& dst_data = node_data.at(dst_node);

            // --- PREPARE LAYOUT ---
            auto layout_builder = DescriptorSetLayoutBuilder();
            uint32_t binding_counter = 0;

            for (auto& input : dst_data.input_connectors) {
                auto& per_input_info = dst_data.input_connections[input];
                std::optional<vk::DescriptorSetLayoutBinding> desc_info =
                    input->get_descriptor_info();
                if (desc_info) {
                    desc_info->setBinding(binding_counter);
                    per_input_info.descriptor_set_binding = binding_counter;
                    layout_builder.add_binding(desc_info.value());

                    binding_counter++;
                }
            }
            for (auto& output : dst_data.output_connectors) {
                auto& per_output_info = dst_data.output_connections[output];
                std::optional<vk::DescriptorSetLayoutBinding> desc_info =
                    output->get_descriptor_info();
                if (desc_info) {
                    desc_info->setBinding(binding_counter);
                    per_output_info.descriptor_set_binding = binding_counter;
                    layout_builder.add_binding(desc_info.value());

                    binding_counter++;
                }
            }
            dst_data.descriptor_set_layout = layout_builder.build_layout(context);
            SPDLOG_DEBUG("descriptor set layout for node {} ({}):\n{}", dst_data.identifier,
                         registry.node_name(dst_node), dst_data.descriptor_set_layout);

            // --- FIND NUMBER OF SETS ---
            // the lowest number of descriptor sets needed.
            std::vector<uint32_t> num_resources;
            // ... number of resources in the corresponding outputs for own inputs
            for (auto& [dst_input, per_input_info] : dst_data.input_connections) {
                if (!per_input_info.node) {
                    // optional input is not connected
                    continue;
                }
                num_resources.push_back(node_data.at(per_input_info.node)
                                            .output_connections[per_input_info.output]
                                            .resources.size());
            }
            // ... number of resources in own outputs
            for (auto& [_, per_output_info] : dst_data.output_connections) {
                num_resources.push_back(per_output_info.resources.size());
            }

            uint32_t num_sets = std::max(lcm(num_resources), ITERATIONS_IN_FLIGHT);
            // make sure it is at least RING_SIZE to allow updates while iterations are in-flight
            // solve k * num_sets >= RING_SIZE
            const uint32_t k = (ITERATIONS_IN_FLIGHT + num_sets - 1) / num_sets;
            num_sets *= k;

            SPDLOG_DEBUG("needing {} descriptor sets for node {} ({})", num_sets,
                         dst_data.identifier, registry.node_name(dst_node));

            // --- ALLOCATE POOL ---
            dst_data.descriptor_pool =
                std::make_shared<DescriptorPool>(dst_data.descriptor_set_layout, num_sets);

            // --- ALLOCATE SETS and PRECOMUTE RESOURCES for each iteration ---
            for (uint32_t set_idx = 0; set_idx < num_sets; set_idx++) {
                // allocate
                dst_data.descriptor_sets.emplace_back(
                    std::make_shared<DescriptorSet>(dst_data.descriptor_pool));

                // precompute resources for inputs
                for (auto& [input, per_input_info] : dst_data.input_connections) {
                    if (!per_input_info.node) {
                        // optional input not connected
                        per_input_info.precomputed_resources.emplace_back(nullptr, -1ul);
                        // apply desc update for optional input here
                        if (dst_data.input_connections[input].descriptor_set_binding !=
                            NodeData::NO_DESCRIPTOR_BINDING) {
                            input->get_descriptor_update(
                                dst_data.input_connections[input].descriptor_set_binding, nullptr,
                                dst_data.descriptor_sets.back(), resource_allocator);
                        }
                    } else {
                        NodeData& src_data = node_data.at(per_input_info.node);
                        assert(src_data.errors.empty());
                        assert(!src_data.disable);
                        auto& resources =
                            src_data.output_connections.at(per_input_info.output).resources;
                        const uint32_t num_resources = resources.size();
                        const uint32_t resource_index =
                            (set_idx + num_resources - input->delay) % num_resources;
                        auto& resource = resources[resource_index];
                        resource.other_set_indices.emplace_back(dst_node, input, set_idx);
                        per_input_info.precomputed_resources.emplace_back(resource.resource,
                                                                          resource_index);
                    }
                }
                // precompute resources for outputs
                for (auto& [_, per_output_info] : dst_data.output_connections) {
                    const uint32_t resource_index = set_idx % per_output_info.resources.size();
                    auto& resource = per_output_info.resources[resource_index];
                    resource.set_indices.emplace_back(set_idx);
                    per_output_info.precomputed_resources.emplace_back(resource.resource,
                                                                       resource_index);
                }

                // precompute resource maps
                dst_data.resource_maps.emplace_back(
                    [&, set_idx](const InputConnectorHandle& connector) {
                        // we set it to null if an optional input is not connected.
                        return std::get<0>(
                            dst_data.input_connections[connector].precomputed_resources[set_idx]);
                    },
                    [&, set_idx](const OutputConnectorHandle& connector) {
                        return std::get<0>(
                            dst_data.output_connections[connector].precomputed_resources[set_idx]);
                    },
                    [&](const OutputConnectorHandle& connector) {
                        return !dst_data.output_connections[connector].inputs.empty();
                    },
                    [&, dst_node]() -> std::any& {
                        return ring_fences.get().user_data.in_flight_data[dst_node];
                    },
                    [&, dst_node](const std::string& event_name, const GraphEvent::Data& data,
                                  const bool notify_all) {
                        send_event(GraphEvent::Info{dst_node, registry.node_name(dst_node),
                                                    dst_data.identifier, event_name},
                                   data, notify_all);
                    });
            }
        }
    }

    std::string make_error_input_not_connected(const InputConnectorHandle& input,
                                               const NodeHandle& node,
                                               const NodeData& data) {
        return fmt::format("the non-optional input {} on node {} ({}) is not "
                           "connected.",
                           input->name, data.identifier, registry.node_name(node));
    }

    void register_event_listener_for_connect(const std::string& event_pattern,
                                             const GraphEvent::Listener& event_listener) {
        split(event_pattern, ",", [&](const std::string& split_pattern) {
            std::smatch match;
            if (!std::regex_match(split_pattern, match, EVENT_REGEX)) {
                SPDLOG_WARN("invalid event pattern '{}'", split_pattern);
                return;
            }
            const std::string& node_name = match[1];
            const std::string& node_identifier = match[2];
            const std::string& event_name = match[3];

            bool registered = false;
            if (node_name.empty()) {
                registered = true;
                if (node_identifier.empty()) {
                    event_listeners["user"][event_name].emplace_back(event_listener);
                    event_listeners["graph"][event_name].emplace_back(event_listener);
                } else if (node_identifier == "user" || node_identifier == "graph") {
                    event_listeners[node_identifier][event_name].emplace_back(event_listener);
                } else {
                    registered = false;
                }
            }
            for (const auto& [identifier, node] : node_for_identifier) {
                if ((node_name.empty() || registry.node_name(node) == node_name) &&
                    (node_identifier.empty() || identifier == node_identifier)) {
                    event_listeners[identifier][event_name].emplace_back(event_listener);
                    registered = true;
                }
            }

            if (registered) {
                SPDLOG_DEBUG("registered listener for event pattern '{}'", split_pattern);
            } else {
                SPDLOG_WARN("no listener registered for event pattern '{}'. (no node type and node "
                            "identifier matched)",
                            split_pattern);
            }
        });
    }

    void send_graph_event(const std::string& event_name,
                          const GraphEvent::Data& data = {},
                          const bool notify_all = true) {
        send_event(GraphEvent::Info{nullptr, "", "graph", event_name}, data, notify_all);
    }

    void send_event(const GraphEvent::Info& event_info,
                    const GraphEvent::Data& data,
                    const bool notify_all) const {
        assert(!event_info.event_name.empty() && "event name cannot be empty.");
        assert(!event_info.identifier.empty() && "identifier cannot be empty.");
        assert((event_info.event_name.find('/') == event_info.event_name.npos) &&
               "event name cannot contain '/'.");

        SPDLOG_TRACE("sending event: {}/{}/{}, notify all={}", event_info.node_name,
                     event_info.identifier, event_info.event_name, notify_all);

        const auto identifier_it = event_listeners.find(event_info.identifier);
        if (identifier_it == event_listeners.end()) {
            return;
        }

        // exact match
        const auto event_it = identifier_it->second.find(event_info.event_name);
        if (event_it != identifier_it->second.end()) {
            if (notify_all) {
                for (const auto& listener : event_it->second) {
                    listener(event_info, data);
                }
            } else {
                for (const auto& listener : event_it->second) {
                    if (listener(event_info, data)) {
                        break;
                    }
                }
            }
        }

        // any
        const auto event_any_it = identifier_it->second.find("");
        if (event_any_it != identifier_it->second.end()) {
            if (notify_all) {
                for (const auto& listener : event_any_it->second) {
                    listener(event_info, data);
                }
            } else {
                for (const auto& listener : event_any_it->second) {
                    if (listener(event_info, data)) {
                        break;
                    }
                }
            }
        }
    }

  public:
    // --- Callback setter ---

    // Set a callback that is executed right after nodes are preprocessed and before any node is
    // run.
    void set_on_run_starting(const std::function<void(GraphRun& graph_run)>& on_run_starting) {
        this->on_run_starting = on_run_starting;
    }

    // Set a callback that is executed right before the commands for this run are submitted to
    // the GPU.
    void set_on_pre_submit(const std::function<void(GraphRun& graph_run)>& on_pre_submit) {
        this->on_pre_submit = on_pre_submit;
    }

    // Set a callback that is executed right after the run was submitted to the queue and the
    // run callbacks were called.
    void set_on_post_submit(const std::function<void()>& on_post_submit) {
        this->on_post_submit = on_post_submit;
    }

  private:
    // General stuff
    const ContextHandle context;
    const ResourceAllocatorHandle resource_allocator;
    const QueueHandle queue;
    std::shared_ptr<ExtensionVkDebugUtils> debug_utils = nullptr;

    NodeRegistry registry;
    ThreadPoolHandle thread_pool;
    CPUQueueHandle cpu_queue;

    // Outside callbacks
    // clang-format off
    std::function<void(GraphRun& graph_run)>                                on_run_starting = [](GraphRun&) {};
    std::function<void(GraphRun& graph_run)>                                on_pre_submit = [](GraphRun&) {};
    std::function<void()>                                                   on_post_submit = [] {};
    // clang-format on

    // Per-iteration data management
    merian::RingFences<ITERATIONS_IN_FLIGHT, InFlightData> ring_fences;

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

} // namespace merian_nodes
