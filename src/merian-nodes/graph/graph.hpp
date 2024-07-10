#pragma once

#include "errors.hpp"
#include "graph_run.hpp"
#include "node.hpp"
#include "resource.hpp"

#include "merian-nodes/graph/node_registry.hpp"
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

    NodeData(const std::string& name) : name(name) {}

    // A unique name for this node from the user. This is not the name from the node registry.
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

        resource_maps.clear();
        descriptor_sets.clear();
        descriptor_pool.reset();
        descriptor_set_layout.reset();

        statistics = {};

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
          registry(context, resource_allocator), ring_fences(context) {
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

    NodeRegistry& get_registry() {
        return registry;
    }

    // Adds a node to the graph.
    //
    // The node_type must be a known type to the registry.
    //
    // Throws invalid_argument, if a node with this name already exists, the graph contains the
    // same node under a different name.
    std::string add_node(const std::string& node_type,
                         const std::optional<std::string>& name = std::nullopt) {
        return add_node(registry.create_node_from_name(node_type), name);
    }

    // Returns nullptr if the node does not exist.
    NodeHandle get_node_for_name(const std::string& name) const {
        if (!node_for_name.contains(name)) {
            return nullptr;
        }
        return node_for_name.at(name);
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
        const NodeHandle src_node = get_node_for_name(src);
        const NodeHandle dst_node = get_node_for_name(dst);
        assert(src_node);
        assert(dst_node);
        add_connection(src_node, dst_node, src_output, dst_input);
    }

    bool remove_connection(const std::string& src,
                           const std::string& dst,
                           const std::string& dst_input) {
        const NodeHandle src_node = get_node_for_name(src);
        const NodeHandle dst_node = get_node_for_name(dst);
        assert(src_node);
        assert(dst_node);
        remove_connection(src_node, dst_node, dst_input);
    }

    // Removes a node from the graph.
    // If a run is in progress the removal is queued for the end of the run.
    bool remove_node(const std::string& name) {
        if (!node_for_name.contains(name)) {
            return false;
        }

        const std::function<void()> remove_task = [this, name] {
            wait();

            const NodeHandle node = node_for_name.at(name);
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

            const std::string node_name = std::move(data.name);
            node_data.erase(node);
            node_for_name.erase(name);
            for (uint32_t i = 0; i < RING_SIZE; i++) {
                InFlightData& in_flight_data = ring_fences.get(i).user_data;
                in_flight_data.in_flight_data.erase(node);
            }

            SPDLOG_DEBUG("removed node {} ({})", node_name, registry.node_name(node));
            needs_reconnect = true;
        };

        if (run_in_progress) {
            SPDLOG_DEBUG("schedule removal of node {} for the end of run the current run.", name);
            on_run_finished_tasks.emplace_back(std::move(remove_task));
        } else {
            remove_task();
        }

        return true;
    }

    // --- connect / run graph ---

    // Attempts to connect the graph with the current set of connections
    // May fail with invalid_connection if there is a illegal connection present (a node input does
    // not support the connected output or the graph contains a undelayed cycle). May fail with
    // connection_missing if a node input was not connected. May fail with conenector_error if two
    // input or output connectors have the same name.
    //
    // If this method returns without throwing the graph was successfully connected and can be run
    // using the run() method.
    //
    // the configuration allow to inspect the partial connections as well
    //
    // Usually you need to call this in a loop until "needs_reconnect" is false.
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
                    MERIAN_PROFILE_SCOPE(
                        profiler, fmt::format("{} ({})", data.name, registry.node_name(node)));
                    SPDLOG_DEBUG("on_connected node: {} ({})", data.name, registry.node_name(node));
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
        run_in_progress = true;

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

            run.reset(iteration, iteration % RING_SIZE, profiler, cmd_pool, resource_allocator);

            // While preprocessing nodes can signalize that they need to reconnect as well
            {
                MERIAN_PROFILE_SCOPE(profiler, "Preprocess nodes");
                for (auto& node : flat_topology) {
                    NodeData& data = node_data.at(node);
                    MERIAN_PROFILE_SCOPE(
                        profiler, fmt::format("{} ({})", data.name, registry.node_name(node)));
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
                    debug_utils->cmd_begin_label(cmd, registry.node_name(node));

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
        for (const auto& task : on_run_finished_tasks)
            task();
        on_run_finished_tasks.clear();
        run_in_progress = false;
    }

    // waits until all in-flight iterations have finished
    void wait() {
        ring_fences.wait_all();
    }

    // removes all nodes and connections from the graph.
    void reset() {
        wait();
        node_data.clear();
        node_for_name.clear();
        needs_reconnect = true;
    }

    // Ensures at reconnect at the next run
    void request_reconnect() {
        needs_reconnect = true;
    }

    std::vector<std::string> node_identifiers() {
        std::vector<std::string> nodes;
        for (const auto& [name, node] : node_for_name) {
            nodes.emplace_back(name);
        }
        return nodes;
    }

    void properties(Properties& props) {
        needs_reconnect |= props.config_bool("Rebuild");
        props.st_no_space();
        props.output_text(fmt::format("Current iteration: {}", iteration));

        if (props.is_ui() &&
            props.st_begin_child("edit", "Edit Graph", Properties::ChildFlagBits::FRAMED)) {
            props.st_separate("Add Node");
            props.config_options("new type", new_node_selected, registry.node_names(),
                                 Properties::OptionsStyle::COMBO);
            if (props.config_text("new identifier", new_node_name.size(), new_node_name.data(),
                                  true, "Set an optional name for the node and press enter.") ||
                props.config_bool("Add Node")) {
                std::optional<std::string> optional_identifier;
                if (new_node_name[0]) {
                    optional_identifier = new_node_name.data();
                }
                add_node(registry.node_names()[new_node_selected], optional_identifier);
            }

            const std::vector<std::string> node_ids = node_identifiers();
            props.st_separate("Add Connection");
            props.config_options("connection src", add_connection_selected_src, node_ids,
                                 Properties::OptionsStyle::COMBO);
            std::vector<std::string> src_outputs;
            for (const auto& [output_name, output] :
                 node_data.at(node_for_name.at(node_ids[add_connection_selected_src]))
                     .output_connector_for_name) {
                src_outputs.emplace_back(output_name);
            }
            props.config_options("connection src output", add_connection_selected_src_output,
                                 src_outputs, Properties::OptionsStyle::COMBO);
            props.config_options("connection dst", add_connection_selected_dst, node_ids,
                                 Properties::OptionsStyle::COMBO);
            NodeData& dst_data =
                node_data.at(node_for_name.at(node_ids[add_connection_selected_dst]));
            std::vector<std::string> dst_inputs;
            for (const auto& [input_name, input] : dst_data.input_connector_for_name) {
                dst_inputs.emplace_back(input_name);
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
                                      it->second.second, node_data.at(it->second.first).name,
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
            props.config_bool("profiling", profiler_enable);
            props.st_no_space();
            props.config_uint("report intervall", profiler_report_intervall_ms,
                              "Set the time period for the profiler to update in ms. Meaning, "
                              "averages and deviations are calculated over this this period.");

            if (profiler_enable) {
                if (last_run_report &&
                    props.st_begin_child("run", "Graph Run",
                                         Properties::ChildFlagBits::DEFAULT_OPEN)) {
                    if (!last_run_report.cpu_report.empty()) {
                        props.st_separate("CPU");
                        props.output_plot_line("",
                                               cpu_time_history.data() + time_history_current + 1,
                                               (cpu_time_history.size() / 2) - 1, 0, cpu_max);
                        props.config_float("cpu max ms", cpu_max, 0, 1000);
                        Profiler::get_cpu_report_as_config(props, last_run_report);
                    }

                    if (!last_run_report.gpu_report.empty()) {
                        props.st_separate("GPU");
                        props.output_plot_line("",
                                               gpu_time_history.data() + time_history_current + 1,
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
            props.st_end_child();
        }

        bool loading = false;
        if (props.st_begin_child("nodes", "Nodes",
                                 Properties::ChildFlagBits::DEFAULT_OPEN |
                                     Properties::ChildFlagBits::FRAMED)) {
            std::vector<std::string> nodes = node_identifiers();

            if (nodes.empty() && !props.is_ui()) {
                nodes = props.st_list_children();

                if (!nodes.empty()) {
                    // go into "loading" mode
                    SPDLOG_INFO(
                        "Attempt to reconstruct the graph from properties. Fingers crossed!");
                    loading = true;
                    reset(); // never know...
                }
            }

            for (const auto& name : nodes) {

                std::string node_label;
                if (!loading) {
                    // otherwise the node data does not exist!
                    const NodeHandle& node = node_for_name.at(name);
                    const auto& data = node_data.at(node);
                    std::string state = "OK";
                    if (data.disable) {
                        state = "DISABLED";
                    } else if (!data.errors.empty()) {
                        state = "ERROR";
                    }

                    node_label =
                        fmt::format("[{}] {} ({})", state, data.name, registry.node_name(node));
                }

                if (props.st_begin_child(name.c_str(), node_label.c_str())) {
                    NodeHandle node;
                    std::string type;

                    // Create Node
                    if (!loading) {
                        node = node_for_name.at(name);
                        type = registry.node_name(node);
                    }
                    props.serialize_string("type", type);
                    if (loading) {
                        node = node_for_name.at(add_node(type, name));
                    }
                    NodeData& data = node_data.at(node);

                    if (props.config_bool("disable", data.disable))
                        request_reconnect();
                    props.st_no_space();
                    if (props.config_bool("Remove")) {
                        remove_node(name);
                    }

                    if (!data.errors.empty()) {
                        props.output_text(
                            fmt::format("Errors:\n  - {}", fmt::join(data.errors, "\n   - ")));
                    }
                    props.st_separate();
                    if (props.st_begin_child("properties", "Properties",
                                             Properties::ChildFlagBits::DEFAULT_OPEN)) {
                        const Node::NodeStatusFlags flags = node->properties(props);
                        needs_reconnect |= flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT;
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
                for (const auto& [node, data] : node_data) {
                    for (const OutgoingNodeConnection& con : data.desired_outgoing_connections) {
                        nlohmann::json j_con;
                        j_con["src"] = data.name;
                        j_con["dst"] = node_data.at(con.dst).name;
                        j_con["src_output"] = con.src_output;
                        j_con["dst_input"] = con.dst_input;

                        connections.push_back(j_con);
                    }
                }
            }
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

    std::string add_node(const std::shared_ptr<Node>& node,
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
                node_name = fmt::format("{} {}", registry.node_name(node), i++);
            } while (node_for_name.contains(node_name));
        }

        node_for_name[node_name] = node;
        node_data.try_emplace(node, node_name);

        needs_reconnect = true;
        SPDLOG_DEBUG("added node {} ({})", node_name, registry.node_name(node));

        return node_name;
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
            const NodeData& old_src_data = node_data.at(old_src);
            SPDLOG_DEBUG("remove conflicting connection {}, {} ({}) -> {}, {} ({})", old_src_output,
                         old_src_data.name, registry.node_name(old_src), dst_input, dst_data.name,
                         registry.node_name(dst));
            remove_connection(old_src, dst, dst_input);
        }

        {
            // outgoing
            const auto [it, inserted] =
                src_data.desired_outgoing_connections.emplace(dst, src_output, dst_input);
            assert(inserted);
        }

        {
            // incoming
            const auto [it, inserted] =
                dst_data.desired_incoming_connections.try_emplace(dst_input, src, src_output);
            assert(inserted);
        }

        needs_reconnect = true;
        SPDLOG_DEBUG("added connection {}, {} ({}) -> {}, {} ({})", src_output, src_data.name,
                     registry.node_name(src), dst_input, dst_data.name, registry.node_name(dst));
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
                        src_data.name, registry.node_name(src), dst_input, dst_data.name,
                        registry.node_name(dst));
            return false;
        }

        const std::string src_output = it->second.second;
        dst_data.desired_incoming_connections.erase(it);

        const auto out_it =
            src_data.desired_outgoing_connections.find({dst, src_output, dst_input});
        // else we did not add the connection properly
        assert(out_it != src_data.desired_outgoing_connections.end());
        src_data.desired_outgoing_connections.erase(out_it);
        SPDLOG_DEBUG("removed connection {}, {} ({}) -> {}, {} ({})", src_output, src_data.name,
                     registry.node_name(src), dst_input, dst_data.name, registry.node_name(dst));

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
                    for (auto& [node, input] : per_output_info.inputs) {
                        receivers.emplace_back(fmt::format("({}, {} ({}))", input->name,
                                                           node_data.at(node).name,
                                                           registry.node_name(node)));
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
                                node_data.at(per_input_info.node).name,
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

        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd,
                                 fmt::format("{} ({})", data.name, registry.node_name(node)));

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
    [[nodiscard]]
    bool cache_node_input_connectors() {
        for (auto& [node, data] : node_data) {
            // Cache input connectors in node_data and check that there are no name conflicts.
            try {
                data.input_connectors = node->describe_inputs();
            } catch (const graph_errors::node_error& e) {
                data.errors.push_back(e.what());
            }
            for (const InputConnectorHandle& input : data.input_connectors) {
                if (data.input_connector_for_name.contains(input->name)) {
                    throw graph_errors::connector_error{
                        fmt::format("node {} contains two input connectors with the same name {}",
                                    registry.node_name(node), input->name)};
                } else {
                    data.input_connector_for_name[input->name] = input;
                }
            }
        }

        // Store connectors that might be connected (there may still be an invalid connection...)
        for (auto& [node, data] : node_data) {
            for (const auto& connection : data.desired_outgoing_connections) {
                NodeData& dst_data = node_data.at(connection.dst);
                if (!dst_data.errors.empty()) {
                    SPDLOG_WARN("node {} has errors and connection {}, {} ({}) -> {}, {} ({}) "
                                "cannot be validated.",
                                dst_data.name, connection.src_output, data.name,
                                registry.node_name(node), connection.dst_input, dst_data.name,
                                registry.node_name(connection.dst));
                    continue;
                }
                if (!dst_data.input_connector_for_name.contains(connection.dst_input)) {
                    SPDLOG_ERROR("node {} ({}) does not have an input {}. Connection is removed.",
                                 dst_data.name, registry.node_name(connection.dst),
                                 connection.dst_input);
                    remove_connection(node, connection.dst, connection.dst_input);
                    return false;
                }
                if (connection.dst == node &&
                    dst_data.input_connector_for_name.at(connection.dst_input)->delay == 0) {
                    // eliminate self loops
                    SPDLOG_ERROR("undelayed (edges with delay = 0) selfloop {} -> {} detected on "
                                 "node {}! Removing connection.",
                                 data.name, connection.src_output, connection.dst_input);
                    remove_connection(node, connection.dst, connection.dst_input);
                    return false;
                }

                const InputConnectorHandle& dst_input =
                    dst_data.input_connector_for_name[connection.dst_input];
                const auto [it, inserted] = maybe_connected_inputs.try_emplace(dst_input, node);

                assert(inserted); // uniqueness should be made sure in add_connection!
            }
        }

        return true;
    }

    // Only for a "satisfied node". Means, all inputs are connected, or delayed or optional and will
    // not be connected.
    void cache_node_output_connectors(const NodeHandle& node, NodeData& data) {
        try {
            data.output_connectors =
                node->describe_outputs(ConnectorIOMap([&](const InputConnectorHandle& input) {
#ifndef NDEBUG
                    if (input->delay > 0) {
                        throw std::runtime_error{fmt::format(
                            "Node {} tried to access an output connector that is connected "
                            "through a delayed input {} (which is not allowed).",
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
                }));
        } catch (const graph_errors::node_error& e) {
            data.errors.emplace_back(std::move(e.what()));
        }

        for (const auto& output : data.output_connectors) {
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
                             data.name, registry.node_name(node), connection.src_output);
                remove_connection(node, connection.dst, connection.dst_input);
                return false;
            }
            const OutputConnectorHandle src_output =
                data.output_connector_for_name[connection.src_output];
            NodeData& dst_data = node_data.at(connection.dst);
            if (dst_data.disable) {
                SPDLOG_DEBUG("skipping connection to disabled node {}, {} ({})",
                             connection.dst_input, dst_data.name,
                             registry.node_name(connection.dst));
                continue;
            }
            if (!dst_data.errors.empty()) {
                SPDLOG_WARN("skipping connection to erroneous node {}, {} ({})",
                            connection.dst_input, dst_data.name,
                            registry.node_name(connection.dst));
                continue;
            }
            if (!dst_data.input_connector_for_name.contains(connection.dst_input)) {
                // since the node is not disabled and not in error state we know the inputs are
                // valid.
                SPDLOG_ERROR("node {} ({}) does not have an input {}. Removing connection.",
                             dst_data.name, registry.node_name(connection.dst),
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
                             data.name, node_data.at(connection.dst).name);
                remove_connection(node, connection.dst, connection.dst_input);
                return false;
            }

            if (!src_output->supports_delay && dst_input->delay > 0) {
                SPDLOG_ERROR("input connector {} of node {} ({}) was connected to output "
                             "connector {} on node {} ({}) with delay {}, however the output "
                             "connector does not support delay. Removing connection.",
                             dst_input->name, dst_data.name, registry.node_name(connection.dst),
                             src_output->name, data.name, registry.node_name(node),
                             dst_input->delay);
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
                SPDLOG_DEBUG("node {} ({}) is disabled, skipping...", data.name,
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
                    SPDLOG_DEBUG("connecting {} ({})", data.name, registry.node_name(node));

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
                                data.errors.emplace_back(
                                    make_error_input_not_connected(input, node, data));
                                break;
                            }
                        } else {
                            NodeData::PerInputInfo& input_info = data.input_connections[input];
                            if (input_info.node && !node_data.at(input_info.node).errors.empty()) {
                                data.input_connections[input] = NodeData::PerInputInfo();
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
                                         src_output->name, src_data.name,
                                         registry.node_name(src_node), dst_input->name,
                                         dst_data.name, registry.node_name(dst_node), e.what());
                            remove_connection(src_node, dst_node, dst_input->name);
                            return false;
                        }
                        it++;
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
                             max_delay + 1, output->name, data.name, registry.node_name(node));
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
            SPDLOG_DEBUG("descriptor set layout for node {} ({}):\n{}", dst_data.name,
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

            uint32_t num_sets = std::max(lcm(num_resources), RING_SIZE);
            // make sure it is at least RING_SIZE to allow updates while iterations are in-flight
            // solve k * num_sets >= RING_SIZE
            const uint32_t k = (RING_SIZE + num_sets - 1) / num_sets;
            num_sets *= k;

            SPDLOG_DEBUG("needing {} descriptor sets for node {} ({})", num_sets, dst_data.name,
                         registry.node_name(dst_node));

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
                           input->name, data.name, registry.node_name(node));
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

    NodeRegistry registry;

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
    bool run_in_progress = false;
    std::vector<std::function<void()>> on_run_finished_tasks;

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

    // Properties helper
    int new_node_selected = 0;
    std::array<char, 128> new_node_name = {0};
    int remove_node_selected = 0;
    int add_connection_selected_src = 0;
    int add_connection_selected_src_output = 0;
    int add_connection_selected_dst = 0;
    int add_connection_selected_dst_input = 0;
};

} // namespace merian_nodes
