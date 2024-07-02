#pragma once

#include "errors.hpp"
#include "graph_run.hpp"
#include "node.hpp"
#include "resource.hpp"

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/sync/ring_fences.hpp"
#include "merian/vk/utils/math.hpp"
#include <merian/vk/descriptors/descriptor_set_layout_builder.hpp>

#include <cstdint>
#include <queue>
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

// Data that is stored for every node that is present in the graph.
struct NodeData {
    static const uint32_t NO_DESCRIPTOR_BINDING = -1u;

    NodeData(const std::string& name) : name(name) {}

    // A unique name for this node from the user. This is not node->name().
    // (on add_node)
    std::string name;

    // User disabled
    bool disable{};
    // Disabled because a input is not connected;
    std::vector<std::string> errors{};

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
    std::unordered_set<NodeConnection, typename NodeConnection::Hash> desired_connections;

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
    struct PerDescriptorSetInfo {
        DescriptorSetHandle descriptor_set;
        std::unique_ptr<DescriptorSetUpdate> update;
    };
    std::vector<PerDescriptorSetInfo> descriptor_sets;
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

        descriptor_sets.clear();
        descriptor_pool.reset();
        descriptor_set_layout.reset();

        errors.clear();
    }
};

inline std::string format_as(const NodeData::NodeStatistics stats) {
    return fmt::format("Descriptor bindings updated: {}", stats.last_descriptor_set_updates);
}

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
    // Data that is stored for every iteration in flight.
    // Created for each iteration in flight in Graph::Graph.
    struct InFlightData {
        // The command pool for the current iteration.
        // We do not use RingCommandPool here since we might want to add a more custom
        // setup later (multi-threaded, multi-queues,...).
        std::shared_ptr<CommandPool> command_pool;
        // Staging set, to release staging buffers and images when the copy
        // to device local memory has finished.
        merian::StagingMemoryManager::SetID staging_set_id{};
        // The graph run, holds semaphores and such.
        GraphRun graph_run{RING_SIZE};
        // Query pools for the profiler
        QueryPoolHandle<vk::QueryType::eTimestamp> profiler_query_pool;
        // Tasks that should be run in the current iteration after acquiring the fence.
        std::vector<std::function<void()>> tasks;
        // For each node: optional in-flight data.
        std::unordered_map<NodeHandle, std::any> in_flight_data{};
    };

  public:
    Graph(const SharedContext& context, const ResourceAllocatorHandle& resource_allocator)
        : context(context), resource_allocator(resource_allocator), queue(context->get_queue_GCT()),
          ring_fences(context) {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            InFlightData& in_flight_data = ring_fences.get(i).user_data;
            in_flight_data.command_pool = std::make_shared<CommandPool>(queue);
            in_flight_data.profiler_query_pool =
                std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context, 512, true);
        }
        debug_utils = context->get_extension<ExtensionVkDebugUtils>();
        run_profiler = std::make_shared<merian::Profiler>(context);
    }

    ~Graph() {
        wait();
    }

    // --- add / remove nodes and connections ---

    // Adds a node to the graph.
    //
    // Throws invalid_argument, if a node with this name already exists, the graph contains the
    // same node under a different name or the name is "Node <number>" which is reserved for
    // internal use.
    void add_node(const std::shared_ptr<Node>& node,
                  const std::optional<std::string>& name = std::nullopt) {
        if (node_data.contains(node)) {
            throw std::invalid_argument{
                fmt::format("graph already contains this node as '{}'", node_data.at(node).name)};
        }

        std::string node_name;
        if (name) {
            if (name->empty()) {
                throw std::invalid_argument{"node name cannot be empty"};
            }
            if (node_for_name.contains(name.value())) {
                throw std::invalid_argument{
                    fmt::format("graph already contains a node with name '{}'", name.value())};
            }
            node_name = name.value();
        } else {
            uint32_t i = 0;
            do {
                node_name = fmt::format("{} {}", node->name, i++);
            } while (node_for_name.contains(node_name));
        }

        node_for_name[node_name] = node;
        node_data.try_emplace(node, node_name);

        needs_reconnect = true;
        SPDLOG_DEBUG("added node {} ({})", node_name, node->name);
    }

    // Adds a connection to the graph.
    //
    // Throws invalid_argument if one of the node does not exist in the graph.
    // The connection is validated on connect(). This means if you want to validate the connection
    // make sure to call connect() as well.
    void add_connection(const NodeHandle& src,
                        const NodeHandle& dst,
                        const std::string& src_output,
                        const std::string& dst_input) {
        if (!node_data.contains(src) || !node_data.contains(dst)) {
            throw std::invalid_argument{"graph does not contain the source or destination node"};
        }

        node_data.at(src).desired_connections.emplace(dst, src_output, dst_input);
        needs_reconnect = true;
    }

    // --- connect / run graph ---

    // Attempts to connect the graph with the current set of connections
    // May fail with illegal_connection if there is a illegal connection present (a node input does
    // not support the connected output or the graph contains a undelayed cycle). May fail with
    // connection_missing if a node input was not connected. May fail with conenector_error if two
    // input or output connectors have the same name.
    //
    // If this method returns without throwing the graph was successfully connected and can be run
    // using the run() method.
    //
    // the configuration allow to inspect the partial connections as well
    void connect() {
        ProfilerHandle profiler = std::make_shared<Profiler>(context);
        {
            MERIAN_PROFILE_SCOPE(profiler, "connect");

            needs_reconnect = false;

            // no nodes -> no connect necessary
            if (node_data.empty()) {
                return;
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
                flat_topology = connect_nodes();
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
                    MERIAN_PROFILE_SCOPE(profiler, fmt::format("{} ({})", data.name, node->name));
                    SPDLOG_DEBUG("on_connected node: {} ({})", data.name, node->name);
                    Node::NodeStatusFlags flags = node->on_connected(data.descriptor_set_layout);
                    needs_reconnect |= flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT;
                    if (flags & Node::NodeStatusFlagBits::RESET_IN_FLIGHT_DATA) {
                        for (uint32_t i = 0; i < RING_SIZE; i++) {
                            ring_fences.get(i).user_data.in_flight_data.at(node).reset();
                        }
                    }
                }
            }
        }
        iteration = 0;
        last_build_report = profiler->get_report();
    }

    // Runs one iteration of the graph.
    //
    // If necessary, the graph is automatically built.
    // The execution is blocked until the fence according to the current iteration is signaled.
    // Interaction with the run is possible using the callbacks.
    void run() {
        // PREPARE RUN: wait for fence, release resources, reset cmd pool

        // wait for the in-flight processing to finish
        InFlightData& in_flight_data = ring_fences.next_cycle_wait_get();

        // now we can release the resources from staging space and reset the command pool
        resource_allocator->getStaging()->releaseResourceSet(in_flight_data.staging_set_id);
        const std::shared_ptr<CommandPool>& cmd_pool = in_flight_data.command_pool;
        GraphRun& run = in_flight_data.graph_run;
        cmd_pool->reset();

        const vk::CommandBuffer cmd = cmd_pool->create_and_begin();
        // get profiler and reports
        const ProfilerHandle& profiler = prepare_profiler_for_run(in_flight_data);

        // CONNECT and PREPROCESS
        do {
            // While connection nodes can signalize that they need to reconnect
            while (needs_reconnect) {
                connect();
            }

            run.reset(iteration, iteration % RING_SIZE, profiler, cmd_pool);

            // While preprocessing nodes can signalize that they need to reconnect as well
            {
                MERIAN_PROFILE_SCOPE(profiler, "Preprocess nodes");
                for (auto& node : flat_topology) {
                    NodeData& data = node_data.at(node);
                    MERIAN_PROFILE_SCOPE(profiler, fmt::format("{} ({})", data.name, node->name));
                    const uint32_t set_idx = iteration % data.descriptor_sets.size();
                    Node::NodeStatusFlags flags =
                        node->pre_process(run, data.resource_maps[set_idx]);
                    needs_reconnect |= flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT;
                    if (flags & Node::NodeStatusFlagBits::RESET_IN_FLIGHT_DATA) {
                        in_flight_data.in_flight_data[node].reset();
                    }
                }
            }
        } while (needs_reconnect);

        // RUN
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "Run nodes");
            on_run_starting(run);
            for (auto& node : flat_topology) {
                NodeData& data = node_data.at(node);
                if (debug_utils)
                    debug_utils->cmd_begin_label(cmd, node->name);

                run_node(run, cmd, node, data, profiler);

                if (debug_utils)
                    debug_utils->cmd_end_label(cmd);
            }
        }

        // FINISH RUN: submit

        on_pre_submit(run, cmd);
        cmd_pool->end_all();
        in_flight_data.staging_set_id = resource_allocator->getStaging()->finalizeResourceSet();
        queue->submit(cmd_pool, ring_fences.reset(), run.get_signal_semaphores(),
                      run.get_wait_semaphores(), run.get_wait_stages(),
                      run.get_timeline_semaphore_submit_info());
        run.execute_callbacks(queue);
        on_post_submit();

        needs_reconnect |= run.needs_reconnect;
        iteration++;
    }

    // waits until all in-flight iterations have finished
    void wait() {
        ring_fences.wait_all();
    }

    // Ensures at reconnect at the next run
    void request_reconnect() {
        needs_reconnect = true;
    }

    void properties(Properties& props) {
        needs_reconnect |= props.config_bool("Rebuild");

        props.output_text(fmt::format("Current iteration: {}", iteration));

        props.st_separate("Profiler");
        props.config_bool("profiling", profiler_enable);
        props.st_no_space();
        props.config_uint("report intervall", profiler_report_intervall_ms,
                          "Set the time period for the profiler to update in ms. Meaning, "
                          "averages and deviations are calculated over this this period.");

        if (profiler_enable) {
            if (last_run_report && props.st_begin_child("run", "Graph Run")) {
                if (!last_run_report.cpu_report.empty()) {
                    props.st_separate("CPU");
                    props.output_plot_line("", cpu_time_history.data() + time_history_current + 1,
                                           (cpu_time_history.size() / 2) - 1, 0, cpu_max);
                    props.config_float("cpu max ms", cpu_max, 0, 1000);
                    Profiler::get_cpu_report_as_config(props, last_run_report);
                }

                if (!last_run_report.gpu_report.empty()) {
                    props.st_separate("GPU");
                    props.output_plot_line("", gpu_time_history.data() + time_history_current + 1,
                                           (gpu_time_history.size() / 2) - 1, 0, gpu_max);
                    props.config_float("gpu max ms", gpu_max, 0, 1000);
                    Profiler::get_gpu_report_as_config(props, last_run_report);
                }
                props.st_end_child();
            }
            if (last_build_report && props.st_begin_child("build", "Last Graph Build")) {
                Profiler::get_report_as_config(props, last_build_report);
                props.st_end_child();
            }
        }

        props.st_separate("Nodes");

        for (const auto& [name, node] : node_for_name) {
            auto& data = node_data.at(node);

            std::string node_label;
            std::string state = "OK";
            if (data.disable) {
                state = "DISABLED";
            } else if (!data.errors.empty()) {
                state = "ERROR";
            }

            node_label = fmt::format("[{}] {} ({})", state, data.name, node->name);

            if (props.st_begin_child(data.name.c_str(), node_label.c_str())) {
                if (props.config_bool("disable node", data.disable))
                    request_reconnect();
                if (!data.errors.empty()) {
                    props.output_text(
                        fmt::format("Errors:\n  - {}", fmt::join(data.errors, "\n   - ")));
                }

                const Node::NodeStatusFlags flags = node->properties(props);
                needs_reconnect |= flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT;
                props.st_separate();
                if (props.st_begin_child("stats", "Statistics")) {
                    props.output_text(fmt::format("{}", data.statistics));
                    props.st_end_child();
                };
                io_props_for_node(props, data);
                props.st_end_child();
            }
        }
    }

  private:
    void io_props_for_node(Properties& config, NodeData& data) {
        if (data.descriptor_set_layout &&
            config.st_begin_child("desc_set_layout", "Descriptor Set Layout")) {
            config.output_text(fmt::format("{}", data.descriptor_set_layout));
            config.st_end_child();
        }
        if (!data.output_connections.empty() && config.st_begin_child("outputs", "Outputs")) {
            for (auto& [output, per_output_info] : data.output_connections) {
                if (config.st_begin_child(output->name, output->name)) {
                    std::vector<std::string> receivers;
                    for (auto& [node, input] : per_output_info.inputs) {
                        receivers.emplace_back(fmt::format("({}, {} ({}))", input->name,
                                                           node_data.at(node).name, node->name));
                    }

                    const uint32_t set_idx = iteration % data.descriptor_sets.size();
                    auto& [_, cur_resource_index] = per_output_info.precomputed_resources[set_idx];

                    config.output_text(fmt::format(
                        "Descriptor set binding: {}\n# Resources: {:02}\nResource index: "
                        "{:02}\nSending to: [{}]",
                        per_output_info.descriptor_set_binding == NodeData::NO_DESCRIPTOR_BINDING
                            ? "None"
                            : std::to_string(per_output_info.descriptor_set_binding),
                        per_output_info.resources.size(), cur_resource_index,
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
        if (!data.input_connections.empty() && config.st_begin_child("inputs", "Inputs")) {
            for (auto& [input, per_input_info] : data.input_connections) {
                if (config.st_begin_child(input->name, input->name)) {
                    config.output_text(fmt::format(
                        "Descriptor set binding: {}",
                        per_input_info.descriptor_set_binding == NodeData::NO_DESCRIPTOR_BINDING
                            ? "None"
                            : std::to_string(per_input_info.descriptor_set_binding)));
                    if (per_input_info.output) {
                        config.output_text(fmt::format(
                            "Receiving from: {}, {} ({})", per_input_info.output->name,
                            node_data.at(per_input_info.node).name, per_input_info.node->name));
                    } else {
                        config.output_text("Optional input not connected.");
                    }
                    config.st_separate("Input Properties");
                    input->properties(config);
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
            return nullptr;
        }

        auto report = run_profiler->set_collect_get_every(in_flight_data.profiler_query_pool,
                                                          profiler_report_intervall_ms);

        if (report) {
            last_run_report = std::move(*report);

            const float cpu_sum = std::transform_reduce(
                last_run_report.cpu_report.begin(), last_run_report.cpu_report.end(), 0,
                std::plus<>(), [](auto& report) { return report.duration; });
            const float gpu_sum = std::transform_reduce(
                last_run_report.gpu_report.begin(), last_run_report.gpu_report.end(), 0,
                std::plus<>(), [](auto& report) { return report.duration; });

            const uint32_t half_size = cpu_time_history.size() / 2;
            cpu_time_history[time_history_current] =
                cpu_time_history[time_history_current + half_size] = cpu_sum;
            gpu_time_history[time_history_current] =
                gpu_time_history[time_history_current + half_size] = gpu_sum;
            time_history_current = (time_history_current + 1) % half_size;
        }

        return run_profiler;
    }

    // Calls connector callbacks, checks resource states and records as well as applies descriptor
    // set updates.
    void run_node(GraphRun& run,
                  const vk::CommandBuffer& cmd,
                  const NodeHandle& node,
                  NodeData& data,
                  [[maybe_unused]] const ProfilerHandle& profiler) {
        const uint32_t set_idx = iteration % data.descriptor_sets.size();

        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, fmt::format("{} ({})", data.name, node->name));

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
                    run, cmd, resource, node, image_barriers, buffer_barriers);
                if (flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) {
                    NodeData& src_data = node_data.at(per_input_info.node);
                    record_descriptor_updates(src_data, per_input_info.output,
                                              src_data.output_connections[per_input_info.output],
                                              resource_index);
                }
                if (flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) {
                    request_reconnect();
                }
            }
            for (auto& [output, per_output_info] : data.output_connections) {
                auto& [resource, resource_index] = per_output_info.precomputed_resources[set_idx];
                const Connector::ConnectorStatusFlags flags = output->on_pre_process(
                    run, cmd, resource, node, image_barriers, buffer_barriers);
                if (flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) {
                    record_descriptor_updates(data, output, per_output_info, resource_index);
                }
                if (flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) {
                    request_reconnect();
                }
            }

            if (!image_barriers.empty() || !buffer_barriers.empty()) {
                vk::DependencyInfoKHR dep_info{{}, {}, buffer_barriers, image_barriers};
                cmd.pipelineBarrier2(dep_info);
                image_barriers.clear();
                buffer_barriers.clear();
            }
        }

        auto& descriptor_set = data.descriptor_sets[set_idx];
        {
            // apply descriptor set updates
            data.statistics.last_descriptor_set_updates = descriptor_set.update->count();
            if (!descriptor_set.update->empty()) {
                SPDLOG_TRACE("applying {} descriptor set updates for node {}, set {}",
                             descriptor_set.update->count(), data.name, set_idx);
                descriptor_set.update->update(context);
                descriptor_set.update->next();
            }
        }

        { node->process(run, cmd, descriptor_set.descriptor_set, data.resource_maps[set_idx]); }

        {
            // Call connector callbacks (post_process) and record descriptor set updates
            for (auto& [input, per_input_info] : data.input_connections) {
                if (!per_input_info.node) {
                    // optional input not connected
                    continue;
                }

                auto& [resource, resource_index] = per_input_info.precomputed_resources[set_idx];
                const Connector::ConnectorStatusFlags flags = input->on_post_process(
                    run, cmd, resource, node, image_barriers, buffer_barriers);
                if (flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) {
                    NodeData& src_data = node_data.at(per_input_info.node);
                    record_descriptor_updates(src_data, per_input_info.output,
                                              src_data.output_connections[per_input_info.output],
                                              resource_index);
                }
                if (flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) {
                    request_reconnect();
                }
            }
            for (auto& [output, per_output_info] : data.output_connections) {
                auto& [resource, resource_index] = per_output_info.precomputed_resources[set_idx];
                const Connector::ConnectorStatusFlags flags = output->on_post_process(
                    run, cmd, resource, node, image_barriers, buffer_barriers);
                if (flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) {
                    record_descriptor_updates(data, output, per_output_info, resource_index);
                }
                if (flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) {
                    request_reconnect();
                }
            }

            if (!image_barriers.empty() || !buffer_barriers.empty()) {
                vk::DependencyInfoKHR dep_info{{}, {}, buffer_barriers, image_barriers};
                cmd.pipelineBarrier2(dep_info);
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
                    *src_data.descriptor_sets[set_idx].update, resource_allocator);
            }

        for (auto& [dst_node, dst_input, set_idx] : resource_info.other_set_indices) {
            NodeData& dst_data = node_data.at(dst_node);
            NodeData::PerInputInfo& per_input_info = dst_data.input_connections[dst_input];
            if (per_input_info.descriptor_set_binding != NodeData::NO_DESCRIPTOR_BINDING)
                dst_input->get_descriptor_update(
                    per_input_info.descriptor_set_binding, resource_info.resource,
                    *dst_data.descriptor_sets[set_idx].update, resource_allocator);
        }
    }

    // --- Graph connect sub-tasks ---

    // Removes all connections, frees graph resources and resets the precomputed topology.
    // Only keeps desired connections.
    void reset_connections() {
        SPDLOG_DEBUG("reset connections");

        this->flat_topology.clear();
        this->maybe_connected_inputs.clear();
        for (auto& [node, data] : node_data) {
            data.reset();
        }
    }

    // Calls the describe_inputs() methods of the nodes and caches the result in
    // - node_data[].input_connectors
    // - node_data[].input_connector_for_name
    // - maybe_connected_inputs
    //
    // For all desired_connections: Ensures that the inputs exists (outputs are NOT checked)
    // and that the connections to all inputs are unique.
    void cache_node_input_connectors() {
        for (auto& [node, data] : node_data) {
            // Cache input connectors in node_data and check that there are no name conflicts.
            data.input_connectors = node->describe_inputs();
            for (const InputConnectorHandle& input : data.input_connectors) {
                if (data.input_connector_for_name.contains(input->name)) {
                    throw graph_errors::connector_error{
                        fmt::format("node {} contains two input connectors with the same name {}",
                                    node->name, input->name)};
                } else {
                    data.input_connector_for_name[input->name] = input;
                }
            }
        }

        // Store connectors that might be connected (there may still be an invalid connection...)
        for (auto& [node, data] : node_data) {
            for (const auto& connection : data.desired_connections) {
                NodeData& dst_data = node_data.at(connection.dst);
                if (!dst_data.input_connector_for_name.contains(connection.dst_input)) {
                    throw graph_errors::illegal_connection{
                        fmt::format("node {} ({}) does not have an input {}.", dst_data.name,
                                    connection.dst->name, connection.dst_input)};
                }

                const InputConnectorHandle& dst_input =
                    dst_data.input_connector_for_name[connection.dst_input];
                const auto [it, inserted] = maybe_connected_inputs.try_emplace(dst_input, node);

                if (!inserted) {
                    throw graph_errors::illegal_connection{
                        fmt::format("the input {} on node ({}) {} is already connected.",
                                    connection.dst_input, data.name, node->name)};
                }
            }
        }
    }

    // Only for a "satisfied node". Means, all inputs are connected, or delayed or optional and will
    // not be connected.
    void cache_node_output_connectors(const NodeHandle& node, NodeData& data) {
        data.output_connectors =
            node->describe_outputs(ConnectorIOMap([&](const InputConnectorHandle& input) {
#ifndef NDEBUG
                if (input->delay > 0) {
                    throw std::runtime_error{
                        fmt::format("Node {} tried to access an output connector that is connected "
                                    "through a delayed input {} (which is not allowed).",
                                    node->name, input->name)};
                }
                if (std::find(data.input_connectors.begin(), data.input_connectors.end(), input) ==
                    data.input_connectors.end()) {
                    throw std::runtime_error{
                        fmt::format("Node {} tried to get an output connector for an input {} "
                                    "which was not returned in describe_inputs (which is not "
                                    "how this works).",
                                    node->name, input->name)};
                }
#endif
                // for optional inputs we inserted a input connection with nullptr in
                // start_nodes, no problem here.
                return data.input_connections.at(input).output;
            }));
        for (const auto& output : data.output_connectors) {
            if (data.output_connector_for_name.contains(output->name)) {
                throw graph_errors::connector_error{
                    fmt::format("node {} contains two output connectors with the same name {}",
                                node->name, output->name)};
            }
            data.output_connector_for_name.try_emplace(output->name, output);
            data.output_connections.try_emplace(output);
        }
    }

    void connect_node(const NodeHandle& node,
                      NodeData& data,
                      const std::unordered_set<NodeHandle>& visited) {
        assert(visited.contains(node) && "necessary for self loop check");

        for (const NodeConnection& connection : data.desired_connections) {
            NodeData& dst_data = node_data.at(connection.dst);
            if (!data.output_connector_for_name.contains(connection.src_output)) {
                throw graph_errors::illegal_connection{
                    fmt::format("node {} ({}) does not have an output {}.", data.name, node->name,
                                connection.src_output)};
            }
            const OutputConnectorHandle src_output =
                data.output_connector_for_name[connection.src_output];
            if (!dst_data.input_connector_for_name.contains(connection.dst_input)) {
                throw graph_errors::illegal_connection{
                    fmt::format("node {} ({}) does not have an input {}.", dst_data.name,
                                connection.dst->name, connection.dst_input)};
            }
            const InputConnectorHandle dst_input =
                dst_data.input_connector_for_name[connection.dst_input];

            // made sure in cache_node_input_connectors
            assert(!dst_data.input_connections.contains(dst_input));

            if (dst_input->delay == 0 && visited.contains(connection.dst)) {
                // this includes self loops because we insert the current node into visited before
                // calling this method. Back-edges with delay > 1 are allowed!
                throw graph_errors::illegal_connection{
                    fmt::format("undelayed (edges with delay = 0) graph is not "
                                "acyclic! {} -> {}",
                                data.name, node_data.at(connection.dst).name)};
            }

            if (!src_output->supports_delay && dst_input->delay > 0) {
                throw graph_errors::illegal_connection{
                    fmt::format("input connector {} of node {} ({}) was connected to output "
                                "connector {} on node {} ({}) with delay {}, however the output "
                                "connector does not support delay.",
                                dst_input->name, dst_data.name, connection.dst->name,
                                src_output->name, data.name, node->name, dst_input->delay)};
            }

            dst_data.input_connections.try_emplace(dst_input,
                                                   NodeData::PerInputInfo{node, src_output});
            data.output_connections[src_output].inputs.emplace_back(connection.dst, dst_input);
        }
    }

    // Helper for topological visit that calculates the next topological layer from the 'not yet
    // visited' cadidate nodes.
    //
    // - Sets disable_missing_input flag if a required input is not connected. In this case the node
    // is removed from candidates.
    //
    // This is used to initialize a topological traversal of the graph to connect the nodes.
    void search_satisfied_nodes(std::set<NodeHandle>& candidates,
                                std::priority_queue<NodeHandle>& queue) {
        std::vector<NodeHandle> to_erase;
        // find nodes with all inputs conencted, delayed, or optional and will not be connected
        for (const NodeHandle& node : candidates) {
            NodeData& data = node_data.at(node);

            if (data.disable) {
                SPDLOG_DEBUG("node {} ({}) is disabled, skipping...", data.name, node->name);
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
                    if (!input->optional) {
                        // This is bad. No node will connect to this input and the input is not
                        // optional...
                        const std::string error = make_error_input_not_connected(input, node, data);
                        SPDLOG_WARN(error);
                        data.errors.emplace_back(std::move(error));

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
    std::vector<NodeHandle> connect_nodes() {
        SPDLOG_DEBUG("connecting nodes");

        cache_node_input_connectors();

        std::vector<NodeHandle> topology;
        topology.reserve(node_data.size());

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
                visited.insert(queue.top());
                NodeData& data = node_data.at(queue.top());

                {
                    assert(!data.disable && data.errors.empty());
                    SPDLOG_DEBUG("connecting {} ({})", data.name, node->name);

                    topology.emplace_back(node);

                    // 1. Get node output connectors and check for name conflicts
                    cache_node_output_connectors(node, data);

                    // 2. Connect outputs to the inputs of destination nodes (fill in their
                    // input_connections and the current nodes output_connections).
                    connect_node(node, data, visited);
                }

                queue.pop();
            }
        }

        // Now it might be possible that a node later in the topolgy was disabled and thus the
        // backward edge does not exist. Therefore we need to traverse the topology and disable
        // those nodes iteratively. Multiple times since disabled nodes, can have backward edges
        // themselfes...
        {
            std::vector<NodeHandle> filtered_topology;
            filtered_topology.reserve(topology.size());

            for (bool changed = true; changed;) {
                changed = false;
                filtered_topology.clear();

                for (const auto& node : topology) {
                    NodeData& data = node_data.at(node);
                    assert(!data.disable);
                    for (const auto& input : data.input_connectors) {
                        if (!data.input_connections.contains(input)) {
                            if (input->optional) {
                                data.input_connections.try_emplace(input, NodeData::PerInputInfo());
                            } else {
                                data.errors.emplace_back(
                                    make_error_input_not_connected(input, node, data));
                                break;
                            }
                        } else {
                            NodeData::PerInputInfo& input_info = data.input_connections[input];
                            if (input_info.node && !node_data.at(input_info.node).errors.empty()) {
                                data.input_connections.try_emplace(input, NodeData::PerInputInfo());
                                break;
                            }
                        }
                    }

                    if (data.errors.empty()) {
                        filtered_topology.emplace_back(node);
                    } else {
                        changed = true;
                    }
                }

                std::swap(filtered_topology, topology);
            };
        }

        // Now clean up this mess. All output connections going to disabled nodes must be
        // eliminated. And finally also call the connector callbacks.
        for (const auto& src_node : topology) {
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
                        per_output_info.inputs.erase(it);
                    } else {
                        src_output->on_connect_input(dst_input);
                        dst_input->on_connect_output(src_output);
                        it++;
                    }
                }
            }
        }

        return topology;
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
                             max_delay + 1, output->name, data.name, node->name);
                for (uint32_t i = 0; i <= max_delay; i++) {
                    const GraphResourceHandle res =
                        output->create_resource(per_output_info.inputs, resource_allocator,
                                                resource_allocator, i, RING_SIZE);
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
            SPDLOG_DEBUG("descriptor set layout for node {} ({}): {}", dst_data.name,
                         dst_node->name, dst_data.descriptor_set_layout);

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

            uint32_t num_sets = std::max(lcm(num_resources), RING_SIZE);
            // make sure it is at least RING_SIZE to allow updates while iterations are in-flight
            // solve k * num_sets >= RING_SIZE
            const uint32_t k = (RING_SIZE + num_sets - 1) / num_sets;
            num_sets *= k;

            SPDLOG_DEBUG("needing {} descriptor sets for node {} ({})", num_sets, dst_data.name,
                         dst_node->name);

            // --- ALLOCATE POOL ---
            dst_data.descriptor_pool =
                std::make_shared<DescriptorPool>(dst_data.descriptor_set_layout, num_sets);

            // --- ALLOCATE SETS and PRECOMUTE RESOURCES for each iteration ---
            for (uint32_t set_idx = 0; set_idx < num_sets; set_idx++) {
                // allocate
                const DescriptorSetHandle desc_set =
                    std::make_shared<DescriptorSet>(dst_data.descriptor_pool);
                dst_data.descriptor_sets.emplace_back();
                dst_data.descriptor_sets.back().descriptor_set = desc_set;
                dst_data.descriptor_sets.back().update =
                    std::make_unique<DescriptorSetUpdate>(desc_set);

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
                                *dst_data.descriptor_sets.back().update, resource_allocator);
                        }
                    } else {
                        NodeData& src_data = node_data.at(per_input_info.node);
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
                    [&, dst_node]() -> std::any& {
                        return ring_fences.get().user_data.in_flight_data[dst_node];
                    });
            }
        }
    }

    std::string make_error_input_not_connected(const InputConnectorHandle& input,
                                               const NodeHandle& node,
                                               const NodeData& data) {
        return fmt::format("the non-optional input {} on node {} ({}) is not "
                           "connected.",
                           input->name, data.name, node->name);
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
    void set_on_pre_submit(const std::function<void(GraphRun& graph_run,
                                                    const vk::CommandBuffer& cmd)>& on_pre_submit) {
        this->on_pre_submit = on_pre_submit;
    }

    // Set a callback that is executed right after the run was submitted to the queue and the
    // run callbacks were called.
    void set_on_post_submit(const std::function<void()>& on_post_submit) {
        this->on_post_submit = on_post_submit;
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
    merian::RingFences<RING_SIZE, InFlightData> ring_fences;

    // State
    bool needs_reconnect = false;
    uint64_t iteration = 0;
    bool profiler_enable = true;
    uint32_t profiler_report_intervall_ms = 50;

    Profiler::Report last_build_report;
    Profiler::Report last_run_report;
    // in ms
    std::array<float, 256> cpu_time_history;
    std::array<float, 256> gpu_time_history;
    float cpu_max = 20, gpu_max = 20;
    // Always write at cpu_time_history_current and cpu_time_history_current + (size >> 1)
    uint32_t time_history_current = 0;
    merian::ProfilerHandle run_profiler;

    // Nodes
    std::map<std::string, NodeHandle> node_for_name;
    std::unordered_map<NodeHandle, NodeData> node_data;
    // After connect() contains the nodes as far as a connection was possible in topological
    // order
    std::vector<NodeHandle> flat_topology;
    // Store connectors that might be connected in start_nodes.
    // There may still be an invalid connection or an outputing node might be actually disabled.
    std::unordered_map<InputConnectorHandle, NodeHandle> maybe_connected_inputs;
};

} // namespace merian_nodes
