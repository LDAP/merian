#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/sync/ring_fences.hpp"

#include "graph_run.hpp"
#include "node.hpp"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace merian_nodes {

using namespace merian;

/**
 * @brief      A Vulkan processing graph.
 *
 * @tparam     RING_SIZE  Controls the amount of in-flight processing (frames-in-flight).
 */
template <uint32_t RING_SIZE = 2>
class Graph : public std::enable_shared_from_this<Graph<RING_SIZE>> {

  private:
    // Describes a connection between two connectors of two nodes.
    struct NodeConnection {
        const NodeHandle dst;
        const std::string src_output;
        const std::string dst_input;

        bool operator==(const NodeConnection&) const = default;

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
        NodeData(Node& node, const std::string& name) : node(node), name(name) {}

        // Reference to the node (for performance reasons)
        // Since we are mapping from NodeHandle to NodeData this reference should stay valid.
        // (on add_node)
        Node& node;

        // A unique name for this node from the user. This is not node->name().
        // (on add_node)
        std::string name;

        // Cache input connectors (node->describe_inputs())
        // (on add_node)
        std::unordered_map<std::string, InputConnectorHandle> input_connectors;
        // Cache output connectors (node->describe_outputs())
        // (on conncet)
        std::unordered_map<std::string, OutputConnectorHandle> output_connectors;

        // --- Desired connections. ---
        // Set by the user using the public add_connection method.
        // This information is used by connect() to build the graph
        std::unordered_set<NodeConnection, typename NodeConnection::Hash> desired_connections;

        // --- Actural connections. ---
        // for each input the connected node and the corresponding output connector on the other
        // node (on connect)
        struct PerInputInfo {
            NodeHandle connected_output;
            OutputConnectorHandle connected_output_handle;
        };
        std::unordered_map<InputConnectorHandle, PerInputInfo> input_connections{};
        // for each output the connected nodes and the corresponding input connector on the other
        // node (on connect)
        struct PerOutputInfo {
            GraphResourceHandle resource;
            std::vector<std::tuple<NodeHandle, OutputConnectorHandle>> connected_inputs;
        };
        std::unordered_map<OutputConnectorHandle, PerOutputInfo> image_output_connections{};
    };

  public:
    Graph(const SharedContext& context, const ResourceAllocatorHandle& resource_allocator)
        : context(context), resource_allocator(resource_allocator), queue(context->get_queue_GCT()),
          ring_fences(context) {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            ring_fences.get(i).user_data.command_pool = std::make_shared<CommandPool>(queue);
        }
        set_profiling(true);
    }

    // --- add / remove nodes and connections ---

    void add_node() {}

    void add_connection() {}

    // --- connect / run graph ---

    // attemps to connect the graph with the current set of connections
    // may fail with illegal_connection if there is a illegal connection present.
    // may fail with connection_missing if a node input was not connected.
    //
    // the configuration allow to inspect the partial connections as well
    void connect() {}

    // Runs one iteration of the graph.
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
        run.reset(nullptr);
        on_run_starting(run);
        const vk::CommandBuffer cmd = cmd_pool->create_and_begin();

        // EXECUTE RUN

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

    // Outside callbacks
    // clang-format off
    std::function<void(GraphRun& graph_run)>                                on_run_starting = [](GraphRun&) {};
    std::function<void(GraphRun& graph_run, const vk::CommandBuffer& cmd)>  on_pre_submit = [](GraphRun&, const vk::CommandBuffer&) {};
    std::function<void()>                                                   on_post_submit = [] {};
    // clang-format on

    // Per-iteration data management
    merian::RingFences<RING_SIZE, IterationData> ring_fences;

    // State
    bool rebuild_requested = true;
    uint64_t iteration = 0;

    // Nodes
    std::unordered_map<std::string, NodeHandle> node_for_name;
    std::unordered_map<NodeHandle, NodeData> node_data;
    // After connect() contains the nodes as far as a connection was possible in topological order
    std::vector<NodeHandle> flat_topology;
};

} // namespace merian_nodes
