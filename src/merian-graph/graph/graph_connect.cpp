#include "merian-graph/graph/graph.hpp"

#include <spdlog/spdlog.h>

namespace merian {

void Graph::connect() {
    ProfilerHandle profiler = std::make_shared<Profiler>(context);
    const ScopedDefaultProfiler scoped_default_profiler{profiler};
    {
        MERIAN_PROFILE_SCOPE(profiler, "connect");

        needs_reconnect = false;

        // fresh per build: descriptor caches must not outlive the topology they were made for
        shader_object_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, ring_fences.size());

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

        std::vector<NodeHandle> topology;
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
            if (!connect_nodes(topology)) {
                SPDLOG_WARN(
                    "Connecting nodes failed :( But attempted self healing. Retry, please!");
                needs_reconnect = true;
                return;
            }
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "build layers");
            build_layers(topology);
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "allocate resources");
            allocate_resources();
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "precompute resources");
            precompute_resources();
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "on_connected");
            queue->submit_wait([&](const CommandBufferHandle& cmd) {
                for (const auto& layer : layers)
                    for (const NodeHandle& node : layer.nodes) {
                        for (auto& [output, per_output_info] :
                             node_data.at(node).output_connections) {
                            std::vector<GraphResourceHandle> resources;
                            resources.reserve(per_output_info.resources.size());
                            for (const auto& per_resource : per_output_info.resources) {
                                resources.push_back(per_resource.resource);
                            }
                            output->on_connected(cmd, resources);
                        }
                    }

                for (auto& layer : layers)
                    for (auto& node : layer.nodes) {
                        NodeData& data = node_data.at(node);
                        MERIAN_PROFILE_SCOPE(profiler, fmt::format("{} ({})", data.identifier,
                                                                   registry.node_type_name(node)));
                        SPDLOG_DEBUG("on_connected node: {} ({})", data.identifier,
                                     registry.node_type_name(node));
                        const NodeIOLayout io_layout(this, &data, node, /*allow_delayed*/ true);
                        try {
                            const Node::NodeStatusFlags flags =
                                node->on_connected(NodeConnectedInfo{io_layout, cmd});
                            needs_reconnect |= flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT;
                            if ((flags & Node::NodeStatusFlagBits::RESET_IN_FLIGHT_DATA) != 0u) {
                                for (uint32_t i = 0; i < ring_fences.size(); i++) {
                                    ring_fences.get(i).user_data.in_flight_data.at(node).reset();
                                }
                            }
                            if ((flags & Node::NodeStatusFlagBits::REMOVE_NODE) != 0u) {
                                remove_node(data.identifier);
                            }
                        } catch (const graph_errors::node_error& e) {
                            data.errors_queued.emplace_back(
                                fmt::format("node error: {}", e.what()));
                        } catch (const GLSLShaderCompiler::compilation_failed& e) {
                            data.errors_queued.emplace_back(
                                fmt::format("compilation failed: {}", e.what()));
                        }
                        if (!data.errors_queued.empty()) {
                            SPDLOG_ERROR("on_connected on node '{}' failed:\n - {}",
                                         data.identifier, fmt::join(data.errors_queued, "\n   - "));
                            request_reconnect();
                            SPDLOG_ERROR("emergency reconnect.");
                        }
                    }
            });
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

void Graph::reset_connections() {
    SPDLOG_DEBUG("reset connections");

    layers.clear();
    maybe_connected_inputs.clear();
    for (auto& [node, data] : node_data) {
        data.reset();
    }
    event_listeners.clear();
}

bool Graph::cache_node_input_connectors() {
    for (auto& [node, data] : node_data) {
        // Cache input connectors in node_data and check that there are no name conflicts.
        try {
            auto input_descriptors = node->describe_inputs();
            for (const auto& desc : input_descriptors) {
                data.input_connectors.push_back(desc.connector);
                if (data.input_connector_for_name.contains(desc.name)) {
                    throw graph_errors::connector_error{
                        fmt::format("node {} contains two input connectors with the same name {}",
                                    registry.node_type_name(node), desc.name)};
                }
                data.input_connector_for_name[desc.name] = desc.connector;
                data.input_name_for_connector[desc.connector] = desc.name;
                data.connector_access[desc.connector] = desc.access;
                data.input_delay[desc.connector] = desc.delay;
                data.input_optional[desc.connector] = desc.optional;
                data.bind_field_name[desc.connector] = "in_" + desc.name;
            }
        } catch (const graph_errors::node_error& e) {
            data.errors.emplace_back(fmt::format("node error: {}", e.what()));
        } catch (const GLSLShaderCompiler::compilation_failed& e) {
            data.errors.emplace_back(fmt::format("compilation failed: {}", e.what()));
        }
        if (!data.errors.empty()) {
            SPDLOG_ERROR("node '{}' ({}) failed to describe its inputs and is disabled:\n - {}",
                         data.identifier, registry.node_type_name(node),
                         fmt::join(data.errors, "\n   - "));
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
                            registry.node_type_name(node), connection.dst_input,
                            dst_data.identifier, registry.node_type_name(connection.dst));
                continue;
            }
            if (!dst_data.input_connector_for_name.contains(connection.dst_input)) {
                SPDLOG_ERROR("node {} ({}) does not have an input {}. Connection is removed.",
                             dst_data.identifier, registry.node_type_name(connection.dst),
                             connection.dst_input);
                remove_connection(node, connection.dst, connection.dst_input);
                return false;
            }
            if (connection.dst == node &&
                dst_data.input_delay.at(
                    dst_data.input_connector_for_name.at(connection.dst_input)) == 0) {
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

void Graph::cache_node_output_connectors(const NodeHandle& node, NodeData& data) {
    try {
        auto output_descriptors =
            node->describe_outputs(NodeIOLayout(this, &data, node, /*allow_delayed*/ false));

        for (const auto& desc : output_descriptors) {
#ifndef NDEBUG
            if (!desc.connector) {
                SPDLOG_CRITICAL("node {} ({}) returned nullptr in describe_outputs",
                                data.identifier, registry.node_type_name(node));
                assert(desc.connector && "node returned nullptr in describe_outputs");
            }
#endif
            if (data.output_connector_for_name.contains(desc.name)) {
                throw graph_errors::connector_error{
                    fmt::format("node {} contains two output connectors with the same name {}",
                                registry.node_type_name(node), desc.name)};
            }
            data.output_connectors.push_back(desc.connector);
            data.output_connector_for_name.try_emplace(desc.name, desc.connector);
            data.output_name_for_connector[desc.connector] = desc.name;
            data.output_connections.try_emplace(desc.connector);
            data.connector_access[desc.connector] = desc.access;
            data.bind_field_name[desc.connector] = "out_" + desc.name;
        }
    } catch (const graph_errors::node_error& e) {
        data.errors.emplace_back(fmt::format("node error: {}", e.what()));
    } catch (const GLSLShaderCompiler::compilation_failed& e) {
        data.errors.emplace_back(fmt::format("compilation failed: {}", e.what()));
    }
    if (!data.errors.empty()) {
        SPDLOG_ERROR("node '{}' ({}) failed to describe its outputs and is disabled:\n - {}",
                     data.identifier, registry.node_type_name(node),
                     fmt::join(data.errors, "\n   - "));
    }
}

bool Graph::connect_node(const NodeHandle& node,
                         NodeData& data,
                         const std::unordered_set<NodeHandle>& visited) {
    assert(visited.contains(node) && "necessary for self loop check");
    assert(data.errors.empty() && data.enabled && !data.unsupported);

    for (const OutgoingNodeConnection& connection : data.desired_outgoing_connections) {
        // since the node is not disabled and not in error state we know the outputs are valid.
        if (!data.output_connector_for_name.contains(connection.src_output)) {
            SPDLOG_ERROR("node {} ({}) does not have an output {}. Removing connection.",
                         data.identifier, registry.node_type_name(node), connection.src_output);
            remove_connection(node, connection.dst, connection.dst_input);
            return false;
        }
        const OutputConnectorHandle src_output =
            data.output_connector_for_name[connection.src_output];
        NodeData& dst_data = node_data.at(connection.dst);
        if (!dst_data.enabled || dst_data.unsupported) {
            SPDLOG_DEBUG("skipping connection to disabled node {}, {} ({})", connection.dst_input,
                         dst_data.identifier, registry.node_type_name(connection.dst));
            continue;
        }
        if (!dst_data.errors.empty()) {
            SPDLOG_WARN("skipping connection to erroneous node {}, {} ({})", connection.dst_input,
                        dst_data.identifier, registry.node_type_name(connection.dst));
            continue;
        }
        if (!dst_data.input_connector_for_name.contains(connection.dst_input)) {
            // since the node is not disabled and not in error state we know the inputs are
            // valid.
            SPDLOG_ERROR("node {} ({}) does not have an input {}. Removing connection.",
                         dst_data.identifier, registry.node_type_name(connection.dst),
                         connection.dst_input);
            remove_connection(node, connection.dst, connection.dst_input);
            return false;
        }
        const InputConnectorHandle dst_input =
            dst_data.input_connector_for_name[connection.dst_input];

        // made sure in cache_node_input_connectors
        assert(!dst_data.input_connections.contains(dst_input));

        // self loops should be elimited in cache_node_input_connectors.
        if (dst_data.input_delay.at(dst_input) == 0 && visited.contains(connection.dst)) {
            // Back-edges with delay > 1 are allowed!
            SPDLOG_ERROR("undelayed (edges with delay = 0) graph is not "
                         "acyclic! {} -> {}. Removing arbitraty edge on the cycle.",
                         data.identifier, node_data.at(connection.dst).identifier);
            remove_connection(node, connection.dst, connection.dst_input);
            return false;
        }

        if (!src_output->supports_delay && dst_data.input_delay.at(dst_input) > 0) {
            SPDLOG_ERROR("input connector {} of node {} ({}) was connected to output "
                         "connector {} on node {} ({}) with delay {}, however the output "
                         "connector does not support delay. Removing connection.",
                         connection.dst_input, dst_data.identifier,
                         registry.node_type_name(connection.dst), connection.src_output,
                         data.identifier, registry.node_type_name(node),
                         dst_data.input_delay.at(dst_input));
            remove_connection(node, connection.dst, connection.dst_input);
            return false;
        }

        dst_data.input_connections.try_emplace(dst_input, NodeData::PerInputInfo{node, src_output});
        data.output_connections[src_output].inputs.emplace_back(connection.dst, dst_input);
    }

    return true;
}

void Graph::search_satisfied_nodes(std::set<NodeHandle>& candidates,
                                   std::priority_queue<NodeHandle>& queue) {
    std::vector<NodeHandle> to_erase;
    // find nodes with all inputs conencted, delayed, or optional and will not be connected
    for (const NodeHandle& node : candidates) {
        NodeData& data = node_data.at(node);

        if (!data.enabled || data.unsupported) {
            SPDLOG_DEBUG("node {} ({}) is disabled, skipping...", data.identifier,
                         registry.node_type_name(node));
            to_erase.push_back(node);
            continue;
        }
        if (!data.errors_queued.empty()) {
            SPDLOG_DEBUG("node {} ({}) has queued errors.");
            move_all(data.errors, data.errors_queued);
            data.errors_queued.clear();
        }
        if (!data.errors.empty()) {
            SPDLOG_DEBUG("node {} ({}) is erroneous, skipping...", data.identifier,
                         registry.node_type_name(node));
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

                if (!connecting_node_data.enabled || connecting_node_data.unsupported ||
                    !connecting_node_data.errors.empty()) {
                    will_not_connect = true;
                }
            }

            if (will_not_connect) {
                if (data.input_delay.at(input) > 0) {
                    // Special case: We could remove the node here already since it will
                    // never be fully connected. However we might want to know the outputs
                    // of the node for GUI and technically the node is "satisfied" for a
                    // call to describe_outputs.
                    // Note: We cannot set the error here since that would lead to other nodes
                    // not connecting other inputs.
                } else if (!data.input_optional.at(input)) {
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
                    data.input_connections.try_emplace(input,
                                                       graph_internal::NodeData::PerInputInfo());
                }
            } else {
                // Something will connect to this node, eventually.
                // We can process this node if the input is either delayed or already connected
                satisfied &=
                    (data.input_connections.contains(input)) || (data.input_delay.at(input) > 0);
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

bool Graph::connect_nodes(std::vector<NodeHandle>& topology) {
    SPDLOG_DEBUG("connecting nodes");

    if (!cache_node_input_connectors()) {
        return false;
    }

    assert(topology.empty());
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
            queue.pop();

            visited.insert(node);
            NodeData& data = node_data.at(node);

            {
                assert(data.enabled && !data.unsupported && data.errors.empty());
                SPDLOG_DEBUG("connecting {} ({})", data.identifier, registry.node_type_name(node));

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

                topology.emplace_back(node);
            }
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
                assert(data.enabled && !data.unsupported);
                for (const auto& input : data.input_connectors) {
                    if (!data.input_connections.contains(input)) {
                        if (data.input_optional.at(input)) {
                            data.input_connections.try_emplace(input, NodeData::PerInputInfo());
                        } else {
                            // Not connected delayed inputs are filtered here.
                            std::string error = make_error_input_not_connected(input, node, data);
                            data.errors.emplace_back(error);
                            SPDLOG_WARN(error);
                        }
                    } else {
                        NodeData::PerInputInfo& input_info = data.input_connections[input];
                        if (input_info.node && !node_data.at(input_info.node).errors.empty()) {
                            if (data.input_optional.at(input)) {
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

            std::swap(filtered_topology, topology);
        };
    }

    // Now clean up this mess. All output connections going to disabled nodes must be
    // eliminated. And finally also call the connector callbacks.
    for (const auto& src_node : topology) {
        NodeData& src_data = node_data.at(src_node);

        for (auto& [src_output, per_output_info] : src_data.output_connections) {
            for (auto it = per_output_info.inputs.begin(); it != per_output_info.inputs.end();) {

                const auto& [dst_node, dst_input] = *it;
                const auto& dst_data = node_data.at(dst_node);

                const std::string& src_output_name =
                    src_data.output_name_for_connector.at(src_output);
                const std::string& dst_input_name = dst_data.input_name_for_connector.at(dst_input);

                if (!dst_data.errors.empty()) {
                    SPDLOG_TRACE("cleanup output connection to erroneous node: {}, {} ({}) -> "
                                 "{}, {} ({})",
                                 src_output_name, src_data.identifier,
                                 registry.node_type_name(src_node), dst_input_name,
                                 dst_data.identifier, registry.node_type_name(dst_node));
                    it = per_output_info.inputs.erase(it);
                } else {
                    try {
                        src_output->on_connect_input(dst_input);
                        dst_input->on_connect_output(src_output);
                    } catch (const graph_errors::invalid_connection& e) {
                        SPDLOG_ERROR("Removing invalid connection {}, {} ({}) -> {}, {} ({}). "
                                     "Reason: {}",
                                     src_output_name, src_data.identifier,
                                     registry.node_type_name(src_node), dst_input_name,
                                     dst_data.identifier, registry.node_type_name(dst_node),
                                     e.what());
                        remove_connection(src_node, dst_node, dst_input_name);
                        return false;
                    }
                    ++it;
                }
            }
        }
    }

    return true;
}

void Graph::allocate_resources() {
    for (const auto& layer : layers)
        for (const auto& node : layer.nodes) {
            auto& data = node_data.at(node);
            for (auto& [output, per_output_info] : data.output_connections) {
                uint32_t max_delay = 0;
                for (auto& input : per_output_info.inputs) {
                    max_delay = std::max(
                        max_delay,
                        node_data.at(std::get<0>(input)).input_delay.at(std::get<1>(input)));
                }

                const std::string& output_name = data.output_name_for_connector.at(output);

                SPDLOG_DEBUG("creating, connecting and allocating {} resources for output {} on "
                             "node {} ({})",
                             max_delay + 1, output_name, data.identifier,
                             registry.node_type_name(node));
                ConnectorAccess combined_access = data.connector_access.at(output);
                for (const auto& [input_node, input] : per_output_info.inputs) {
                    combined_access =
                        combined_access | node_data.at(input_node).connector_access.at(input);
                }

                for (uint32_t i = 0; i <= max_delay; i++) {
                    const GraphResourceHandle res = output->create_resource(
                        per_output_info.inputs, combined_access, resource_allocator,
                        resource_allocator, i, ring_fences.size());
                    per_output_info.resources.emplace_back(res);
                }
            }
        }
}

void Graph::precompute_resources() {
    for (auto& layer : layers)
        for (auto& dst_node : layer.nodes) {
            auto& dst_data = node_data.at(dst_node);

            // --- FIND NUMBER OF RESOURCE COMBINATIONS ---
            // the lowest number of per-iteration resource maps needed (delayed accesses cycle).
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

            uint32_t num_sets = std::max(lcm(num_resources), ring_fences.size());
            // make sure it is at least RING_SIZE to allow updates while iterations are in-flight
            // solve k * num_sets >= RING_SIZE
            const uint32_t k = (ring_fences.size() + num_sets - 1) / num_sets;
            num_sets *= k;

            SPDLOG_DEBUG("needing {} resource maps for node {} ({})", num_sets, dst_data.identifier,
                         registry.node_type_name(dst_node));

            // --- PRECOMPUTE RESOURCES for each iteration ---
            for (uint32_t set_idx = 0; set_idx < num_sets; set_idx++) {
                // precompute resources for inputs
                for (auto& [input, per_input_info] : dst_data.input_connections) {
                    if (!per_input_info.node) {
                        // optional input not connected
                        per_input_info.precomputed_resources.emplace_back(nullptr, -1ul);
                    } else {
                        NodeData& src_data = node_data.at(per_input_info.node);
                        assert(src_data.errors.empty());
                        assert(src_data.enabled && !src_data.unsupported);
                        auto& resources =
                            src_data.output_connections.at(per_input_info.output).resources;
                        const uint32_t num_resources = resources.size();
                        const uint32_t resource_index =
                            (set_idx + num_resources - dst_data.input_delay.at(input)) %
                            num_resources;
                        auto& resource = resources[resource_index];
                        per_input_info.precomputed_resources.emplace_back(resource.resource,
                                                                          resource_index);
                    }
                }
                // precompute resources for outputs
                for (auto& [_, per_output_info] : dst_data.output_connections) {
                    const uint32_t resource_index = set_idx % per_output_info.resources.size();
                    auto& resource = per_output_info.resources[resource_index];
                    per_output_info.precomputed_resources.emplace_back(resource.resource,
                                                                       resource_index);
                }

                // precompute resource maps
                dst_data.resource_maps.emplace_back(this, &dst_data, dst_node, set_idx);
            }
        }
}

void Graph::build_layers(const std::vector<NodeHandle>& topology) {
    // dependency layers; delayed inputs read last iteration's resource and impose no order
    uint32_t layer_count = 0;
    for (const NodeHandle& node : topology) {
        NodeData& data = node_data.at(node);
        data.level = 0;
        for (const auto& [input, per_input_info] : data.input_connections) {
            if (!per_input_info.node || data.input_delay.at(input) > 0 ||
                per_input_info.node == node) {
                continue;
            }
            data.level = std::max(data.level, node_data.at(per_input_info.node).level + 1);
        }
        layer_count = std::max(layer_count, data.level + 1);
    }

    layers.assign(layer_count, {});
    for (const NodeHandle& node : topology) {
        layers[node_data.at(node).level].nodes.push_back(node);
    }

    struct Masks {
        vk::PipelineStageFlags2 stages;
        vk::AccessFlags2 access;
        vk::AccessFlags2 write_access;
    };
    std::vector<Masks> masks(layer_count);
    Masks all{};
    for (uint32_t l = 0; l < layer_count; l++) {
        for (const NodeHandle& node : layers[l].nodes) {
            for (const auto& [connector, access] : node_data.at(node).connector_access) {
                masks[l].stages |= access.stages;
                masks[l].access |= access.access;
                if (access.is_write()) {
                    masks[l].write_access |= access.access;
                }
            }
        }
        all.stages |= masks[l].stages;
        all.access |= masks[l].access;
        all.write_access |= masks[l].write_access;
    }

    // src covers all previous layers cumulatively (a layer-1 write read in layer 4 must be
    // covered; execution dependencies also chain into earlier iterations on the same queue).
    // layers[0] orders the whole frame against the previous iteration (delayed inputs, ring
    // reuse).
    if (layer_count > 0) {
        layers[0].barrier =
            vk::MemoryBarrier2{all.stages, all.write_access, all.stages, all.access};
    }
    Masks prefix{};
    for (uint32_t l = 1; l < layer_count; l++) {
        prefix.stages |= masks[l - 1].stages;
        prefix.write_access |= masks[l - 1].write_access;
        layers[l].barrier = vk::MemoryBarrier2{prefix.stages, prefix.write_access, masks[l].stages,
                                               masks[l].access};
    }
}

std::string Graph::make_error_input_not_connected(const InputConnectorHandle& input,
                                                  const NodeHandle& node,
                                                  const NodeData& data) {
    return fmt::format("the non-optional input {} on node {} ({}) is not "
                       "connected.",
                       data.input_name_for_connector.at(input), data.identifier,
                       registry.node_type_name(node));
}

} // namespace merian
