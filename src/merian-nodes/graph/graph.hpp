#pragma once

#include "errors.hpp"
#include "graph_run.hpp"
#include "node.hpp"

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/sync/ring_fences.hpp"

#include <cstdint>
#include <regex>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace merian_nodes {
namespace graph_internal {
// Describes a connection between two connectors of two nodes.
struct NodeConnection {
    const NodeHandle dst;
    const std::string src_output;
    const std::string dst_input;

    bool operator==(const NodeConnection&) const = default;

  public:
    struct Hash {
        size_t operator()(const NodeConnection& c) const noexcept {
            return hash_val(c.dst, c.src_output, c.dst_input);
        }
    };
};

// Data that is stored for every iteration in flight.
// Created for each iteration in flight in Graph::Graph.
struct IterationData {
    // The command pool for the current iteration.
    // We do not use RingCommandPool here since we might want to add a more custom
    // setup later (multi-threaded, multi-queues,...).
    std::shared_ptr<CommandPool> command_pool;
    // Statging set, to release staging buffers and images when the copy
    // to device local memory has finished.
    merian::StagingMemoryManager::SetID staging_set_id{};
    // The graph run, holds semaphores and such.
    GraphRun graph_run;
    // The profiler, might be nullptr if profiling is disabeled.
    merian::ProfilerHandle profiler{};
    // Tasks that should be run in the current iteration after acquiring the fence.
    std::vector<std::function<void()>> tasks;
};

// Data that is stored for every node that is present in the graph.
struct NodeData {
    NodeData(Node& node, const std::string& name, const uint32_t node_number)
        : node(node), name(name), node_number(node_number) {}

    // Reference to the node (for performance reasons)
    // Since we are mapping from NodeHandle to NodeData this reference should stay valid.
    // (on add_node)
    Node& node;

    // A unique name for this node from the user. This is not node->name().
    // (on add_node)
    std::string name;
    const uint32_t node_number;

    // Cache input connectors (node->describe_inputs())
    // (on start_nodes added and checked for name conflicts)
    std::unordered_map<std::string, InputConnectorHandle> input_connectors;
    // Cache output connectors (node->describe_outputs())
    // (on conncet_nodes added and checked for name conflicts)
    std::unordered_map<std::string, OutputConnectorHandle> output_connectors;

    // --- Desired connections. ---
    // Set by the user using the public add_connection method.
    // This information is used by connect() to connect the graph
    std::unordered_set<NodeConnection, typename NodeConnection::Hash> desired_connections;

    // --- Actural connections. ---
    // for each input the connected node and the corresponding output connector on the other
    // node (on connect)
    struct PerInputInfo {
        NodeHandle node;
        OutputConnectorHandle output;
    };
    std::unordered_map<InputConnectorHandle, PerInputInfo> input_connections{};
    // for each output the connected nodes and the corresponding input connector on the other
    // node (on connect)
    struct PerOutputInfo {
        // (max_delay + 1) resources
        std::vector<GraphResourceHandle> resource;
        std::vector<std::tuple<NodeHandle, InputConnectorHandle>> inputs;
    };
    std::unordered_map<OutputConnectorHandle, PerOutputInfo> output_connections{};

    // Precomputed descriptor set layout including all input and output connectors which
    // get_descriptor_info() does not return std::nullopt.
    DescriptorSetLayoutHandle descriptor_set_layout;
};
} // namespace graph_internal

using namespace merian;
using namespace graph_internal;

/**
 * @brief      A Vulkan processing graph.
 *
 * @tparam     RING_SIZE  Controls the amount of in-flight processing (frames-in-flight).
 */
template <uint32_t RING_SIZE = 2>
class Graph : public std::enable_shared_from_this<Graph<RING_SIZE>> {
  public:
    Graph(const SharedContext& context, const ResourceAllocatorHandle& resource_allocator)
        : context(context), resource_allocator(resource_allocator), queue(context->get_queue_GCT()),
          ring_fences(context) {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            ring_fences.get(i).user_data.command_pool = std::make_shared<CommandPool>(queue);
        }
        set_profiling(true);
        debug_utils = context->get_extension<ExtensionVkDebugUtils>();
    }

    // --- add / remove nodes and connections ---

    // Adds a node to the graph.
    //
    // Throws invalid_argument, if a node with this name already exists, the graph contains the
    // same node under a different name or the name is "Node <number>" which is reserved for
    // internal use.
    void add_node(const std::shared_ptr<Node>& node, const std::optional<std::string>& name) {
        if (node_data.contains(node)) {
            throw std::invalid_argument{
                fmt::format("graph already contains this node as '{}'", node_data[node].name)};
        }

        std::string node_name;
        if (name) {
            if (name->empty()) {
                throw std::invalid_argument{"node name cannot be empty"};
            }
            if (std::regex_search(name.value(), std::regex("Node \\d+"))) {
                throw std::invalid_argument{
                    fmt::format("The node name {} is reserved for internal use", name.value())};
            }
            if (node_for_name.contains(name.value())) {
                throw std::invalid_argument{
                    fmt::format("graph already contains a node with name '{}'", name.value())};
            }
        }

        uint32_t node_number;
        if (free_node_numbers.empty()) {
            node_number = this->node_number++;
        } else {
            auto smallest_it = free_node_numbers.begin();
            node_number = *smallest_it;
            free_node_numbers.erase(smallest_it);
        }
        if (node_name.empty()) {
            node_name = fmt::format("Node {}", node_number);
        }

        node_for_name[node_name] = node;
        node_data.emplace(node, *node, node_name, node_number);

        needs_reconnect = true;
        SPDLOG_DEBUG("added node {} {}", node_number, node_name);
    }

    // Adds a connection to the graph.
    //
    // Throws invalid_argument if one of the node does not exist in the graph.
    // The connection is valid on connect().
    void add_connection(const NodeHandle& src,
                        const NodeHandle& dst,
                        const std::string& src_output,
                        const std::string& dst_input) {
        if (!node_data.contains(src) || !node_data.contains(dst)) {
            throw std::invalid_argument{"graph does not contain the source or destination node"};
        }

        node_data[src].desired_connections.emplace(dst, src_output, dst_input);
        needs_reconnect = true;
    }

    // --- connect / run graph ---

    // attemps to connect the graph with the current set of connections
    // May fail with illegal_connection if there is a illegal connection present (a node input does
    // not support the connected output or the graph contains a undelayed cycle). May fail with
    // connection_missing if a node input was not connected. May fail with conenctor_error if two
    // input or output connectors have the same name.
    //
    // If this method returns without throwing the graph was successfully connected and can be run
    // using the run() method.
    //
    // the configuration allow to inspect the partial connections as well
    void connect() {
        // MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Graph: connect");

        needs_reconnect = false;

        // no nodes -> no connect necessary
        if (node_data.empty()) {
            return;
        }

        // Make sure resources are not in use
        queue->wait_idle();

        reset_connections();

        flat_topology = connect_nodes();

        if (flat_topology.size() != node_data.size()) {
            // todo: determine node and input and provide a better error message.
            throw graph_errors::connection_missing{"Graph not fully connected."};
        }

        for (auto& node : flat_topology) {
            [[maybe_unused]] NodeData& data = node_data.at(node);
            // MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, fmt::format("{} ({})", data.name,
            // node->name()));
            // SPDLOG_DEBUG("on_connected node: {} ({})", data.name, node->name);
            // node->on_connected(data.descriptor_set_layout);
        }
    }

    // Runs one iteration of the graph.
    //
    // If necessary, the graph is automatically built.
    // The execution is blocked until the fence according to the current iteration is signaled.
    // Interaction with the run is possible using the callbacks.
    void run() {
        // PREPARE RUN: wait for fence, release resources, reset cmd pool

        // wait for the in-flight processing to finish
        auto& iteration_data = ring_fences.next_cycle_wait_and_get();
        // now we can release the resources from staging space and reset the command pool
        resource_allocator->getStaging()->releaseResourceSet(
            iteration_data.user_data.staging_set_id);
        // run all queued tasks
        std::for_each(iteration_data.user_data.tasks.begin(), iteration_data.user_data.tasks.end(),
                      [](auto& task) { task(); });
        iteration_data.user_data.tasks.clear();
        // get and reset command pool and graph run
        const std::shared_ptr<CommandPool>& cmd_pool = iteration_data.user_data.command_pool;
        cmd_pool->reset();
        GraphRun& run = iteration_data.user_data.graph_run;
        run.reset(nullptr); // todo: profiler
        on_run_starting(run);
        const vk::CommandBuffer cmd = cmd_pool->create_and_begin();

        // MERIAN_PROFILE_SCOPE_GPU(iteration_data.user_data.profiler, cmd, "Graph: run");

        // EXECUTE RUN
        do {
            // While connection nodes can signalize that they need to reconnect
            while (needs_reconnect) {
                connect();
            }
            // While preprocessing nodes can signalize that they need to reconnect as well
            {
                // MERIAN_PROFILE_SCOPE(profiler, "Graph: preprocess nodes");
                for (auto& node : flat_topology) {
                    // MERIAN_PROFILE_SCOPE(profiler, fmt::format("{} ({})", data.name,
                    // node->name()));
                    Node::NodeStatusFlags flags = node->pre_process(run, ConnectorResourceMap());
                    needs_reconnect |= flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT;
                    if (flags & Node::NodeStatusFlagBits::RESET_IN_FLIGHT_DATA) {
                        // todo
                    }
                }
            }
        } while (needs_reconnect);

        {
            // MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Graph: run nodes");
            for (auto& node : flat_topology) {
                // NodeData& data = node_data[node];
                if (debug_utils)
                    debug_utils->cmd_begin_label(cmd, node->name);

                // MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, fmt::format("{} ({})", data.name,
                // node->name())); cmd_run_node(cmd, node, data, graph_frame_data);

                if (debug_utils)
                    debug_utils->cmd_end_label(cmd);
            }
        }

        // FINISH RUN: submit

        on_pre_submit(run, cmd);
        cmd_pool->end_all();
        iteration_data.user_data.staging_set_id =
            resource_allocator->getStaging()->finalizeResourceSet();
        queue->submit(cmd_pool, iteration_data.fence, run.get_signal_semaphores(),
                      run.get_wait_semaphores(), run.get_wait_stages(),
                      run.get_timeline_semaphore_submit_info());
        run.execute_callbacks(queue);
        on_post_submit();

        needs_reconnect = run.needs_reconnect;
        iteration++;
    }

  private:
    // --- Helpers ---

    void add_task_to_current_iteration(const std::function<void()>& task) {
        ring_fences.get().tasks.push_back(task);
    }

    void add_task_to_next_iteration(const std::function<void()>& task) {
        ring_fences.get((ring_fences.current_cycle_index() + 1) % RING_SIZE).tasks.push_back(task);
    }

    void add_task_to_all_iterations_in_flight(const std::function<void()>& task) {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            ring_fences.get(i).user_data.tasks.push_back(task);
        }
    }

    // --- Graph connect subtasks ---

    // Removes all connections, frees graph resources and resets the precomputed topology.
    // Only keeps desired connections.
    void reset_connections() {
        this->flat_topology.clear();
        for (auto& [node, data] : node_data) {

            data.input_connectors.clear();
            data.output_connectors.clear();

            data.input_connections.clear();
            data.output_connections.clear();

            data.descriptor_set_layout.reset();
        }
    }

    // Calls the describe_inputs() methods of the nodes and caches the result in the node_data.
    //
    // Nodes without inputs or with delayed inputs only (ie. nodes that are fully connected). This
    // is used to initialize a topological traversal of the graph to connect the nodes.
    std::queue<NodeHandle> start_nodes() {
        std::queue<NodeHandle> queue;

        for (auto& [node, node_data] : node_data) {
            // Cache input connectors in node_data and check that there are no name conflicts.
            for (InputConnectorHandle& input : node->describe_inputs()) {
                if (node_data.input_connectors.contains(input->name)) {
                    throw graph_errors::connector_error{
                        fmt::format("node {} contains two input connectors with the same name {}",
                                    node->name, input->name)};
                } else {
                    node_data.input_connectors[input->name] = input;
                }
            }

            // Find nodes without inputs or with delayed inputs only.
            if (node_data.input_connectors.empty() ||
                std::all_of(node_data.input_connectors.begin(), node_data.input_connectors.end(),
                            [](auto& input) { return input.second->delay > 0; })) {
                queue.push(node);
            }
        }
        return queue;
    }

    // Returns a topological order of the nodes.
    std::vector<NodeHandle> connect_nodes() {
        std::vector<NodeHandle> topological_order;
        topological_order.reserve(node_data.size());

        topological_visit([&](NodeHandle& node, NodeData& data) {
            topological_order.emplace_back(node);

            // All inputs are connected, i.e. input_connectors and input_connections are valid.
            // That means we can compute the nodes' outputs and fill in inputs
            // of the following nodes.

            // 1. Get node output connectors and check for name conflicts
            std::vector<OutputConnectorHandle> outputs =
                node->describe_outputs(ConnectorIOMap([&](const InputConnectorHandle& input) {
                    if (input->delay > 0) {
                        throw std::runtime_error{fmt::format(
                            "Node {} tried to access an output connector that is connected "
                            "through a delayed input {} (which is not allowed).",
                            node->name, input->name)};
                    }
                    if (!data.input_connections.contains(input)) {
                        throw std::runtime_error{
                            fmt::format("Node {} tried to get an output connector for an input {} "
                                        "which was not returned in describe_inputs (which is not "
                                        "how this works).",
                                        node->name, input->name)};
                    }
                    return data.input_connections.at(input).output;
                }));
            for (auto& output : outputs) {
                if (data.output_connectors.contains(output->name)) {
                    throw graph_errors::connector_error{
                        fmt::format("node {} contains two output connectors with the same name {}",
                                    node->name, output->name)};
                }
            }
        });

        return topological_order;
    }

    // Visites nodes in topological order as far as they are connected or a cycle is detected.
    // Returns the number of visited nodes.
    //
    // Throws if the undelayed graph is not acyclic (feedback edges must have a delay of at
    // least 1).
    void topological_visit(const std::function<void(NodeHandle&, NodeData&)> visitor) {
        std::unordered_set<NodeHandle> visited;
        std::queue<NodeHandle> queue = start_nodes();

        while (!queue.empty()) {
            NodeData& data = node_data.at(queue.front());

            visitor(queue.front(), data);
            visited.insert(queue.front());

            // check for all subsequent nodes if we visited all "requirements" and add to queue.
            // also, fail if we see a node again! (in both cases exclude "feedback" edges)

            // find all subsequent nodes that are connected over a edge with delay = 0 (others are
            // allowed to lie 'behind').
            std::unordered_set<NodeHandle> candidates;
            for (auto& output : data.output_connections) {
                for (auto& [dst_node, image_input] : output.second.inputs) {
                    if (image_input->delay == 0) {
                        candidates.insert(dst_node);
                    }
                }
            }

            // add to queue if all "inputs" were visited
            for (const NodeHandle& candidate : candidates) {
                if (visited.contains(candidate)) {
                    // Back-edges with delay > 1 are allowed!
                    throw graph_errors::illegal_connection{fmt::format(
                        "undelayed (edges with delay = 0) graph is not acyclic! {} -> {}",
                        data.name, node_data.at(candidate).name)};
                }
                bool satisfied = true;
                NodeData& candidate_data = node_data.at(candidate);
                for (auto& candidate_input : candidate_data.input_connectors) {
                    if (candidate_input.second->delay > 0) {
                        // all good, delayed inputs must not be connected.
                        continue;
                    }
                    auto& candidate_input_connection =
                        candidate_data.input_connections.at(candidate_input.second);
                    if (!visited.contains(candidate_input_connection.node)) {
                        // src was not processed, cannot add...
                        satisfied = false;
                        break;
                    }
                }
                if (satisfied) {
                    queue.push(candidate);
                }
            }

            queue.pop();
        }
    }

  public:
    // --- Callback setter ---

    // Set a callback that is executed right after the fence for the current iteration is aquired
    // and before any node is run.
    void set_on_run_starting(const std::function<void(GraphRun& graph_run)>& on_run_starting) {
        this->on_run_starting = on_run_starting;
    }

    // Set a callback that is executed right before the commands for this run are submitted to the
    // GPU.
    void set_on_pre_submit(const std::function<void(GraphRun& graph_run,
                                                    const vk::CommandBuffer& cmd)>& on_pre_submit) {
        this->on_pre_submit = on_pre_submit;
    }

    // Set a callback that is executed right after the run was submitted to the queue and the run
    // callbacks were called.
    void set_on_post_submit(const std::function<void()>& on_post_submit) {
        this->on_post_submit = on_post_submit;
    }

    void set_profiling(const bool enabled) {
        add_task_to_all_iterations_in_flight([&]() {
            if (enabled && !ring_fences.get().user_data.profiler) {
                ring_fences.get().user_data.profiler =
                    std::make_shared<merian::Profiler>(context, queue);
            } else if (!enabled) {
                ring_fences.get().user_data.profiler = nullptr;
            }
        });
    }

  private:
    // General stuff
    const SharedContext context;
    const ResourceAllocatorHandle resource_allocator;
    const QueueHandle queue;
    std::shared_ptr<ExtensionVkDebugUtils> debug_utils = nullptr;

    // Outside callbacks
    // clang-format off
    std::function<void(GraphRun& graph_run)>                                on_run_starting = [](GraphRun&) {};
    std::function<void(GraphRun& graph_run, const vk::CommandBuffer& cmd)>  on_pre_submit = [](GraphRun&, const vk::CommandBuffer&) {};
    std::function<void()>                                                   on_post_submit = [] {};
    // clang-format on

    // Per-iteration data management
    merian::RingFences<RING_SIZE, IterationData> ring_fences;

    // State
    bool needs_reconnect = false;
    uint64_t iteration = 0;

    // Nodes
    std::unordered_map<std::string, NodeHandle> node_for_name;
    std::unordered_map<NodeHandle, NodeData> node_data;
    // After connect() contains the nodes as far as a connection was possible in topological order
    std::vector<NodeHandle> flat_topology;
    // for node naming / identification
    uint32_t node_number = 0;
    std::set<uint32_t> free_node_numbers;
};

} // namespace merian_nodes
