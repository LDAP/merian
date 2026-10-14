#include "merian-graph/graph/graph.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_set>

namespace merian {

void Graph::connect() {
    {
        const BusyScope busy_scope(*this);
        connect_graph();
    }
    apply_deferred_edits();
}

void Graph::connect_graph() {
    ProfilerHandle profiler = std::make_shared<Profiler>(context);
    const ScopedDefaultProfiler scoped_default_profiler{profiler};
    try {
        MERIAN_PROFILE_SCOPE(profiler, "connect");

        needs_reconnect = false;

        // fresh per build: descriptor caches must not outlive the topology they were made for
        shader_object_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, ring_fences.size());

        if (node_data.empty()) {
            reset_connections();
            transient_blocks.clear();
            return;
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "notify nodes");
            send_graph_event("connect");
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "wait for in-flight iterations");
            wait();
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "reset");
            // its descriptor sets hold the resources of the last connect
            run_info.shader_object_allocator.reset();
            reset_connections();
        }

        std::vector<NodeHandle> topology;
        {
            MERIAN_PROFILE_SCOPE(profiler, "connect nodes");
            if (!connect_nodes(topology)) {
                SPDLOG_WARN("connecting nodes failed, connecting again");
                needs_reconnect = true;
                return;
            }
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "build layers");
            build_layers(topology);
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "plan memory");
            plan_memory();
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "precompute resources");
            precompute_resources();
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "on_connected");
            Submission submission(context, queue, cpu_queue);
            // last: their initialization must survive the others sharing their memory
            for (const bool read_before_written : {false, true}) {
                if (read_before_written) {
                    submission.get_cmd()->barrier(vk::MemoryBarrier2{
                        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
                        vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite});
                }
                for (const auto& layer : layers)
                    for (const NodeHandle& node : layer.nodes) {
                        const NodeData& data = node_data.at(node);
                        for (const auto& [output, per_output_info] : data.output_connections) {
                            if (!per_output_info.resource_data || per_output_info.version_of) {
                                continue;
                            }
                            const std::vector<ResourceData::Copy>& copies =
                                per_output_info.resource_data->copies;
                            for (uint32_t copy_index = 0; copy_index < copies.size();
                                 copy_index++) {
                                if (copies[copy_index].read_before_written == read_before_written) {
                                    output->on_connected(
                                        {.allocator = data.allocator, .copy_index = copy_index},
                                        submission, copies[copy_index].resource);
                                }
                            }
                        }
                    }
            }

            for (const auto& layer : layers)
                for (const NodeHandle& node : layer.nodes) {
                    NodeData& data = node_data.at(node);
                    MERIAN_PROFILE_SCOPE(profiler, data.label);
                    const NodeIOLayout io_layout(this, &data, node, /*allow_delayed*/ true);
                    const NodeIO io(this, &data, node, data.set_index(0),
                                    /*allow_transient_outputs*/ false);
                    Node::NodeStatusFlags flags = 0;
                    collect_node_errors(data.errors_queued, [&] {
                        flags = node->on_connected(
                            io_layout, io, NodeConnectionInfo{ring_fences.size()}, submission);
                    });
                    needs_reconnect |= (flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT) != 0u;
                    if ((flags & Node::NodeStatusFlagBits::RESET_IN_FLIGHT_DATA) != 0u) {
                        for (uint32_t i = 0; i < ring_fences.size(); i++) {
                            ring_fences.get(i).user_data.in_flight_data.erase(node);
                        }
                    }
                    if ((flags & Node::NodeStatusFlagBits::REMOVE_NODE) != 0u) {
                        remove_node(data.identifier);
                    }
                    if (!data.errors_queued.empty()) {
                        SPDLOG_ERROR("on_connected of {} failed:\n - {}", data.label,
                                     fmt::join(data.errors_queued, "\n - "));
                        needs_reconnect = true;
                    }
                }
            if (!needs_reconnect) {
                for (auto& [node, data] : node_data) {
                    data.errors_queued.clear();
                }
            }

            submission.finish();
            queue->wait_idle();
        }
    } catch (...) {
        // resources may be half allocated
        needs_reconnect = true;
        throw;
    }

    {
        MERIAN_PROFILE_SCOPE(profiler, "register user event listener");
        for (const auto& [event_pattern, event_listener] : user_event_pattern_listener) {
            register_event_listener_for_connect(event_pattern, event_listener);
        }
    }

    std::vector<std::string> dead_nodes;
    for (const auto& [node, data] : node_data) {
        if (!data.enabled || (data.parent && !node_data.at(data.parent).runs())) {
            continue;
        }
        // on_connected errors are only queued
        const std::vector<std::string>& errors =
            data.errors.empty() ? data.errors_queued : data.errors;
        if (!errors.empty()) {
            dead_nodes.push_back(fmt::format("{}: {}", data.label, fmt::join(errors, "; ")));
        } else if (data.force_disabled) {
            dead_nodes.push_back(fmt::format("{}: {}", data.label, data.force_disabled_reason));
        } else if (data.unsupported) {
            dead_nodes.push_back(fmt::format("{}: {}", data.label, data.unsupported_reason));
        }
    }
    if (!dead_nodes.empty()) {
        std::ranges::sort(dead_nodes);
        SPDLOG_WARN("{} node(s) are NOT running:\n - {}", dead_nodes.size(),
                    fmt::join(dead_nodes, "\n - "));
    }

    run_iteration = 0;
    last_build_report = profiler->get_report();
    time_connect_reference = std::chrono::high_resolution_clock::now();
    duration_elapsed_since_connect = 0ns;
}

void Graph::reset_connections() {
    layers.clear();
    memory_view_dirty = true;
    for (auto& [node, data] : node_data) {
        data.reset();
    }
    event_listeners.clear();
}

namespace {
// Stands for the output of an unconnected optional in-place input.
class AbsentOutput : public OutputConnector {
  public:
    GraphResourceHandle create_resource(const ResourceCreateInfo&) override {
        throw std::logic_error{"an absent output has no resource"};
    }

    bool supports_delay() const override {
        return false;
    }
};

std::optional<std::string>
input_declaration_error(const std::vector<InputConnectorDescriptor>& descriptors) {
    std::unordered_set<std::string> names;
    std::unordered_set<std::string> in_place;
    for (const InputConnectorDescriptor& descriptor : descriptors) {
        if (!names.insert(descriptor.name).second) {
            return fmt::format("two inputs are named {}", descriptor.name);
        }
        if (descriptor.in_place) {
            in_place.insert(descriptor.name);
        }
    }
    std::unordered_set<std::string> passed_on;
    for (const InputConnectorDescriptor& descriptor : descriptors) {
        if (descriptor.in_place && descriptor.delay > 0) {
            return fmt::format("input {} is written in place and delayed", descriptor.name);
        }
        if (descriptor.in_place && !descriptor.access.empty() && !descriptor.access.is_write()) {
            return fmt::format("input {} is written in place but declares only reads",
                               descriptor.name);
        }
        if (descriptor.in_place && !descriptor.pass_through.empty()) {
            return fmt::format("input {} is written in place and passes through", descriptor.name);
        }
        if (descriptor.in_place && !passed_on.insert(descriptor.name).second) {
            return fmt::format("input {} is written in place and passed through to",
                               descriptor.name);
        }
        if (!descriptor.pass_through.empty() && descriptor.delay > 0) {
            return fmt::format("input {} passes through and is delayed", descriptor.name);
        }
        if (!descriptor.pass_through.empty() && !passed_on.insert(descriptor.pass_through).second) {
            return fmt::format("input {} passes through to {}, which another input passes to",
                               descriptor.name, descriptor.pass_through);
        }
        const AutoConnect& auto_connect = descriptor.auto_connect;
        const bool unknown_input =
            auto_connect.from == AutoConnect::From::INPUT &&
            (auto_connect.name == descriptor.name || !names.contains(auto_connect.name) ||
             in_place.contains(auto_connect.name));
        if (auto_connect && (descriptor.delay == 0 || unknown_input)) {
            return fmt::format("input {} has an invalid auto connection", descriptor.name);
        }
    }
    std::unordered_map<std::string, std::string> follows;
    for (const InputConnectorDescriptor& descriptor : descriptors) {
        if (descriptor.auto_connect.from == AutoConnect::From::INPUT) {
            follows.emplace(descriptor.name, descriptor.auto_connect.name);
        }
    }
    for (const auto& [input, followed] : follows) {
        std::string current = followed;
        for (size_t step = 0; step < follows.size(); step++) {
            if (current == input) {
                return fmt::format("input {} is auto connected to itself through other inputs",
                                   input);
            }
            const auto next = follows.find(current);
            if (next == follows.end()) {
                break;
            }
            current = next->second;
        }
    }
    return std::nullopt;
}

bool writes_in_place(const graph_internal::NodeData& data, const std::string& input) {
    const auto connector = data.input_connector_for_name.find(input);
    return connector != data.input_connector_for_name.end() &&
           data.input_descriptors.at(connector->second).in_place;
}
} // namespace

std::optional<graph_internal::Port> Graph::auto_connection_source(
    const NodeHandle& node,
    const NodeData& data,
    std::string input,
    const std::function<std::optional<Port>(const std::string&)>& source_of) {
    for (std::unordered_set<std::string> followed; followed.insert(input).second;) {
        const auto connector = data.input_connector_for_name.find(input);
        if (connector == data.input_connector_for_name.end()) {
            return std::nullopt;
        }
        const AutoConnect& auto_connect = data.input_descriptors.at(connector->second).auto_connect;
        if (auto_connect.from == AutoConnect::From::OUTPUT) {
            return Port{node, auto_connect.name};
        }
        if (auto_connect.from != AutoConnect::From::INPUT) {
            return std::nullopt;
        }
        input = auto_connect.name;
        if (std::optional<Port> source = source_of(input)) {
            return source;
        }
    }
    return std::nullopt;
}

bool Graph::connect_nodes(std::vector<NodeHandle>& topology) {
    // 1. subgraphs, inputs and auto connections
    build_subgraphs();
    cache_node_input_connectors();
    // before the subgraphs: an auto connection can wire a public input
    materialize_auto_connections();
    connect_subgraphs();

    // 2. the user's connections to inputs that do not exist, errors from earlier attempts and runs
    std::vector<Connection> removals;
    for (auto& [node, data] : node_data) {
        // the subgraph of an unsupported node is unknown
        if (data.errors.empty() && !data.unsupported) {
            for (const auto& [input, source] : data.desired_incoming_connections) {
                if (!data.input_connector_for_name.contains(input) &&
                    !data.public_inputs.contains(input)) {
                    SPDLOG_ERROR("{} does not have an input {}. Removing connection.", data.label,
                                 input);
                    removals.push_back({source.node, source.name, node, input});
                }
            }
        }
        data.errors.insert(data.errors.end(), data.errors_queued.begin(), data.errors_queued.end());
        const auto wired = [&](const std::string& input) {
            return wiring(node, input).has_value();
        };
        data.bypassed =
            !data.enabled &&
            (std::ranges::any_of(data.pass_through_inputs,
                                 [&](const auto& entry) { return wired(entry.second); }) ||
             std::ranges::any_of(data.input_descriptors, [&](const auto& entry) {
                 return entry.second.in_place && wired(entry.second.name);
             }));
    }

    // 3. the nodes that run; wiring that cannot work is removed at all nodes
    visit_nodes(topology, removals);
    settle(topology, removals);
    for (auto& [node, data] : node_data) {
        if (data.visited) {
            continue;
        }
        for (const InputConnectorHandle& input : data.input_connectors) {
            if (const InputDecision decision = decide_input(node, data, input);
                decision.kind == InputDecision::Kind::REJECTED && decision.removable) {
                SPDLOG_ERROR("{} Removing connection.", decision.reason);
                removals.push_back(*decision.removable);
            }
        }
    }
    if (!removals.empty()) {
        // inputs auto connected to another one last: removing that one removes them as well, to
        // connect again with it
        std::ranges::stable_partition(removals, [&](const Connection& connection) {
            const NodeData& data = node_data.at(connection.dst);
            const auto input = data.input_connector_for_name.find(connection.dst_input);
            return input == data.input_connector_for_name.end() ||
                   data.input_descriptors.at(input->second).auto_connect.from !=
                       AutoConnect::From::INPUT;
        });
        std::set<std::pair<NodeHandle, std::string>> removed;
        for (const Connection& connection : removals) {
            NodeData& data = node_data.at(connection.dst);
            const auto desired = data.desired_incoming_connections.find(connection.dst_input);
            if (!removed.emplace(connection.dst, connection.dst_input).second ||
                desired == data.desired_incoming_connections.end()) {
                continue;
            }
            const Port source = desired->second;
            if (!remove_connection(connection.src, connection.dst, connection.dst_input)) {
                continue;
            }
            if (const auto input = data.input_connector_for_name.find(connection.dst_input);
                input != data.input_connector_for_name.end() &&
                data.input_descriptors.at(input->second).auto_connect) {
                data.settled_auto_connections.erase(connection.dst_input);
                data.rejected_auto_connections.insert_or_assign(connection.dst_input, source);
            }
        }
        return false;
    }
    connect_outputs(topology);

    // 4. in-place rules
    return validate_in_place_writes(topology);
}

void Graph::cache_node_input_connectors() {
    for (auto& [node, data] : node_data) {
        // its subgraph failed
        if (!data.errors.empty()) {
            continue;
        }
        collect_node_errors(data.errors, [&] {
            const std::vector<InputConnectorDescriptor> descriptors = node->describe_inputs();
            if (const std::optional<std::string> error = input_declaration_error(descriptors)) {
                data.errors.emplace_back(*error);
                return;
            }
            for (const InputConnectorDescriptor& descriptor : descriptors) {
                data.input_connectors.push_back(descriptor.connector);
                data.input_connector_for_name[descriptor.name] = descriptor.connector;
                data.input_descriptors[descriptor.connector] = descriptor;
                data.bind_field_name[descriptor.connector] = "in_" + descriptor.name;
                if (!descriptor.pass_through.empty()) {
                    data.pass_through_inputs[descriptor.pass_through] = descriptor.name;
                }
            }
        });
        if (!data.errors.empty()) {
            SPDLOG_ERROR("{} failed to describe its inputs and is disabled:\n - {}", data.label,
                         fmt::join(data.errors, "\n - "));
        }
    }
}

void Graph::materialize_auto_connections() {
    for (auto& [node, data] : node_data) {
        if (!data.errors.empty() || data.parent) {
            continue;
        }
        for (const InputConnectorHandle& input : data.input_connectors) {
            const InputConnectorDescriptor& descriptor = data.input_descriptors.at(input);
            if (!descriptor.auto_connect ||
                data.settled_auto_connections.contains(descriptor.name)) {
                continue;
            }
            if (!data.desired_incoming_connections.contains(descriptor.name)) {
                const std::optional<Port> source = auto_connection_source(
                    node, data, descriptor.name, [&](const std::string& followed) {
                        const auto connected = data.desired_incoming_connections.find(followed);
                        return connected == data.desired_incoming_connections.end()
                                   ? std::nullopt
                                   : std::optional<Port>(connected->second);
                    });
                const auto rejected = data.rejected_auto_connections.find(descriptor.name);
                // once the followed input is connected, to a source other than the rejected
                if (!source || (rejected != data.rejected_auto_connections.end() &&
                                rejected->second == *source)) {
                    continue;
                }
                add_desired_connection(source->node, node, source->name, descriptor.name);
            }
            data.rejected_auto_connections.erase(descriptor.name);
            data.settled_auto_connections.insert(descriptor.name);
        }
    }
}

std::optional<graph_internal::Connection> Graph::wiring(NodeHandle node, std::string input) const {
    for (;;) {
        const NodeData& data = node_data.at(node);
        if (const auto desired = data.desired_incoming_connections.find(input);
            desired != data.desired_incoming_connections.end()) {
            return Connection{desired->second.node, desired->second.name, node, input};
        }
        if (const auto source = data.subgraph_sources.find(input);
            source != data.subgraph_sources.end()) {
            return Connection{source->second.node, source->second.name, node, input};
        }
        const auto port = data.public_feeds.find(input);
        if (port == data.public_feeds.end()) {
            return std::nullopt;
        }
        input = port->second;
        node = data.parent;
    }
}

graph_internal::Port Graph::follow(Port port) const {
    for (std::set<std::pair<NodeHandle, std::string>> passed;
         passed.emplace(port.node, port.name).second;) {
        const NodeData& data = node_data.at(port.node);
        std::optional<Connection> received;
        if (const auto input = data.pass_through_inputs.find(port.name);
            !data.enabled && input != data.pass_through_inputs.end()) {
            received = wiring(port.node, input->second);
        }
        if (!received) {
            if (const auto child = data.public_outputs.find(port.name);
                child != data.public_outputs.end()) {
                port = child->second;
                continue;
            }
            if (!data.runs() && writes_in_place(data, port.name)) {
                received = wiring(port.node, port.name);
            }
        }
        if (!received) {
            break;
        }
        port = {received->src, received->src_output};
    }
    return port;
}

std::optional<graph_internal::Connection>
Graph::direct_connection(Port consumer, const Connection& wired, const Port& source) const {
    // the wiring of a subgraph is no desired connection
    if (node_data.at(wired.dst).parent) {
        return std::nullopt;
    }
    // an auto connection to a public input is not what the user wired
    while (consumer.node != wired.dst) {
        const NodeData& data = node_data.at(consumer.node);
        const std::string& port = data.public_feeds.at(consumer.name);
        if (std::ranges::find(node_data.at(data.parent).public_inputs.at(port), consumer) ==
            node_data.at(data.parent).public_inputs.at(port).end()) {
            return std::nullopt;
        }
        consumer = {data.parent, port};
    }
    Port port{wired.src, wired.src_output};
    for (auto child = node_data.at(port.node).public_outputs.find(port.name);
         child != node_data.at(port.node).public_outputs.end();
         child = node_data.at(port.node).public_outputs.find(port.name)) {
        port = child->second;
    }
    if (port != source || writes_in_place(node_data.at(source.node), source.name)) {
        return std::nullopt;
    }
    return wired;
}

graph_internal::InputDecision Graph::decide_input(const NodeHandle& node,
                                                  const NodeData& data,
                                                  const InputConnectorHandle& input) const {
    using Kind = InputDecision::Kind;
    const InputConnectorDescriptor& descriptor = data.input_descriptors.at(input);
    const std::optional<Connection> wired = wiring(node, descriptor.name);
    if (!wired) {
        return {.kind = Kind::UNCONNECTED};
    }
    const Port source = follow({wired->src, wired->src_output});
    const NodeData& source_data = node_data.at(source.node);
    const auto reject = [&](const Kind kind, const std::string& reason, const bool removable) {
        const std::optional<Connection> direct =
            removable ? direct_connection({node, descriptor.name}, *wired, source) : std::nullopt;
        if (!direct && !node_data.at(wired->dst).parent) {
            return InputDecision{.kind = kind,
                                 .reason = fmt::format("{} (wired to {}.{})", reason,
                                                       node_data.at(wired->src).identifier,
                                                       wired->src_output)};
        }
        return InputDecision{.kind = kind, .reason = reason, .removable = direct};
    };

    if (source.node == node && descriptor.delay == 0) {
        return reject(Kind::REJECTED,
                      fmt::format("undelayed self loop {} -> {} on {}.", source.name,
                                  descriptor.name, data.label),
                      true);
    }
    if (!source_data.runs()) {
        return {.kind = Kind::STOPPED, .reason = not_running_reason(source.node)};
    }
    if (!source_data.visited) {
        return {.kind = Kind::WAIT, .producer = source.node};
    }
    const auto output = source_data.output_connector_for_name.find(source.name);
    if (output == source_data.output_connector_for_name.end()) {
        // a missing output of a child is an error of its subgraph
        return reject(Kind::REJECTED,
                      fmt::format("{} does not have an output {}.", source_data.label, source.name),
                      !source_data.parent);
    }
    const OutputConnectorDescriptor& output_descriptor =
        source_data.output_descriptors.at(output->second);
    if (output_descriptor.scratch) {
        return reject(Kind::REJECTED,
                      fmt::format("output {} of {} is scratch memory and cannot be connected.",
                                  source.name, source_data.label),
                      true);
    }
    if (output_descriptor.disabled) {
        return {.kind = Kind::STOPPED,
                .reason = fmt::format("{} disabled the output feeding {}", source_data.identifier,
                                      descriptor.name)};
    }
    if (descriptor.delay > 0 && output_descriptor.persistent) {
        return reject(Kind::INCOMPATIBLE,
                      fmt::format("input {} of {} reads output {} of {} with delay {}, but the "
                                  "output is persistent.",
                                  descriptor.name, data.label, source.name, source_data.label,
                                  descriptor.delay),
                      true);
    }
    if (descriptor.delay > 0 && !output->second->supports_delay()) {
        return reject(Kind::INCOMPATIBLE,
                      fmt::format("input {} of {} reads output {} of {} with delay {}, but the "
                                  "output does not support delay.",
                                  descriptor.name, data.label, source.name, source_data.label,
                                  descriptor.delay),
                      true);
    }
    if (!input->can_receive_from(output->second)) {
        return reject(Kind::INCOMPATIBLE,
                      fmt::format("input {} of {} cannot receive from output {} of {}.",
                                  descriptor.name, data.label, source.name, source_data.label),
                      true);
    }
    return {.kind = Kind::CONNECTED, .producer = source.node, .output = output->second};
}

bool Graph::apply_decision(NodeData& data,
                           const InputConnectorHandle& input,
                           const InputDecision& decision,
                           std::vector<Connection>& removals) {
    using Kind = InputDecision::Kind;
    assert(decision.kind != Kind::WAIT);
    const InputConnectorDescriptor& descriptor = data.input_descriptors.at(input);
    if (decision.kind == Kind::REJECTED || decision.kind == Kind::INCOMPATIBLE) {
        if (decision.removable) {
            SPDLOG_ERROR("{} Removing connection.", decision.reason);
            removals.push_back(*decision.removable);
        } else if (descriptor.optional) {
            SPDLOG_WARN("{} Input {} of {} stays unconnected.", decision.reason, descriptor.name,
                        data.label);
        }
    }
    if (decision.kind != Kind::CONNECTED && !descriptor.optional) {
        if (decision.kind == Kind::UNCONNECTED) {
            data.errors.emplace_back(make_error_input_not_connected(input, data));
            SPDLOG_WARN("{}", data.errors.back());
        } else if (decision.kind == Kind::STOPPED) {
            force_disable(data, decision.reason);
        } else {
            SPDLOG_ERROR("{}", decision.reason);
            data.errors.emplace_back(decision.reason);
        }
        data.input_connections.erase(input);
        return true;
    }
    const NodeData::PerInputInfo connected{decision.producer, decision.output};
    const auto [it, inserted] = data.input_connections.try_emplace(input, connected);
    if (inserted) {
        return true;
    }
    if (it->second.node == connected.node && it->second.output == connected.output) {
        return false;
    }
    it->second = connected;
    return true;
}

void Graph::visit_nodes(std::vector<NodeHandle>& topology, std::vector<Connection>& removals) {
    std::map<std::string, NodeHandle> candidates;
    for (const auto& [node, data] : node_data) {
        candidates.try_emplace(data.identifier, node);
    }
    while (!candidates.empty()) {
        // 1. the candidates whose undelayed inputs are decided, in identifier order
        const size_t candidate_count = candidates.size();
        std::vector<NodeHandle> ready;
        std::unordered_map<NodeHandle, std::vector<NodeHandle>> waiting;
        for (auto it = candidates.begin(); it != candidates.end();) {
            const NodeHandle node = it->second;
            NodeData& data = node_data.at(node);
            if (!stop_with_parent(data) && data.runs()) {
                for (const InputConnectorHandle& input : data.input_connectors) {
                    if (data.input_descriptors.at(input).delay > 0 ||
                        data.input_connections.contains(input)) {
                        continue;
                    }
                    const InputDecision decision = decide_input(node, data, input);
                    if (decision.kind == InputDecision::Kind::WAIT) {
                        waiting[node].push_back(decision.producer);
                        continue;
                    }
                    apply_decision(data, input, decision, removals);
                    if (!data.runs()) {
                        break;
                    }
                }
            }
            if (!data.runs()) {
                waiting.erase(node);
                it = candidates.erase(it);
            } else if (!waiting.contains(node)) {
                ready.push_back(node);
                it = candidates.erase(it);
            } else {
                it++;
            }
        }

        // 2. the candidates on a cycle of undelayed inputs
        if (ready.empty() && candidates.size() == candidate_count) {
            for (const NodeHandle& node : waiting_on_themselves(waiting)) {
                NodeData& data = node_data.at(node);
                SPDLOG_ERROR("{} waits on a cycle of connections without delay", data.label);
                data.errors.emplace_back("waits on a cycle of connections without delay");
            }
            continue;
        }

        // 3. their outputs
        for (const NodeHandle& node : ready) {
            NodeData& data = node_data.at(node);
            cache_node_output_connectors(node, data);
            if (data.runs()) {
                data.visited = true;
                topology.push_back(node);
            }
        }
    }
}

std::vector<NodeHandle> Graph::waiting_on_themselves(
    const std::unordered_map<NodeHandle, std::vector<NodeHandle>>& waiting) const {
    std::vector<NodeHandle> on_cycle;
    for (const auto& [node, producers] : waiting) {
        std::unordered_set<NodeHandle> reached;
        for (std::vector<NodeHandle> stack = producers; !stack.empty();) {
            const NodeHandle producer = stack.back();
            stack.pop_back();
            if (!reached.insert(producer).second) {
                continue;
            }
            if (const auto it = waiting.find(producer); it != waiting.end()) {
                stack.insert(stack.end(), it->second.begin(), it->second.end());
            }
        }
        if (reached.contains(node)) {
            on_cycle.push_back(node);
        }
    }
    // every candidate waits on another, so this is not expected
    if (on_cycle.empty()) {
        std::ranges::transform(waiting, std::back_inserter(on_cycle),
                               [](const auto& entry) { return entry.first; });
    }
    return on_cycle;
}

void Graph::settle(std::vector<NodeHandle>& topology, std::vector<Connection>& removals) {
    for (bool changed = true; changed;) {
        changed = false;
        for (const NodeHandle& node : topology) {
            NodeData& data = node_data.at(node);
            if (stop_with_parent(data)) {
                changed = true;
                continue;
            }
            if (!data.runs()) {
                continue;
            }
            // delayed inputs, and inputs whose producer stopped or disabled the output
            for (const InputConnectorHandle& input : data.input_connectors) {
                if (const auto connected = data.input_connections.find(input);
                    connected != data.input_connections.end() &&
                    (!connected->second.node ||
                     (node_data.at(connected->second.node).runs() &&
                      !node_data.at(connected->second.node)
                           .output_descriptors.at(connected->second.output)
                           .disabled))) {
                    continue;
                }
                changed |= apply_decision(data, input, decide_input(node, data, input), removals);
                if (!data.runs()) {
                    break;
                }
            }
            if (!data.runs()) {
                continue;
            }
            // a version whose in-place input is not connected has no resource
            for (const auto& [output, per_output_info] : data.output_connections) {
                OutputConnectorDescriptor& descriptor = data.output_descriptors.at(output);
                if (per_output_info.version_of && !descriptor.disabled &&
                    !data.input_connections.at(per_output_info.version_of).node) {
                    descriptor.disabled = true;
                    changed = true;
                }
            }
        }
    }
    std::erase_if(topology, [&](const NodeHandle& node) { return !node_data.at(node).runs(); });
}

void Graph::connect_outputs(const std::vector<NodeHandle>& topology) {
    for (const NodeHandle& node : topology) {
        const NodeData& data = node_data.at(node);
        for (const InputConnectorHandle& input : data.input_connectors) {
            const NodeData::PerInputInfo& connected = data.input_connections.at(input);
            if (connected.node) {
                node_data.at(connected.node)
                    .output_connections.at(connected.output)
                    .inputs.emplace_back(node, input);
                input->on_connect_output(connected.output);
            }
        }
    }
}

void Graph::force_disable(NodeData& data, const std::string& reason) {
    if (data.force_disabled) {
        return;
    }
    data.force_disabled = true;
    data.force_disabled_reason = reason;
    SPDLOG_INFO("{} force disabled: {}", data.label, reason);
}

bool Graph::stop_with_parent(NodeData& data) {
    if (!data.parent || !data.runs()) {
        return false;
    }
    const std::string reason = not_running_reason(data.parent);
    if (reason.empty()) {
        return false;
    }
    force_disable(data, reason);
    return true;
}

std::string Graph::not_running_reason(const NodeHandle& node) const {
    const NodeData& data = node_data.at(node);
    if (!data.enabled) {
        return fmt::format("{} is disabled", data.identifier);
    }
    if (data.unsupported) {
        return fmt::format("{} is unsupported", data.identifier);
    }
    if (data.force_disabled) {
        return data.force_disabled_reason;
    }
    if (!data.errors.empty()) {
        return fmt::format("{} has an error", data.identifier);
    }
    return {};
}

void Graph::cache_node_output_connectors(const NodeHandle& node, NodeData& data) {
    std::vector<OutputConnectorDescriptor> descriptors;
    collect_node_errors(data.errors, [&] {
        descriptors =
            node->describe_outputs(NodeIOLayout(this, &data, node, /*allow_delayed*/ false));
    });
    const auto add = [&](const OutputConnectorDescriptor& descriptor,
                         const InputConnectorHandle& version_of = nullptr) {
        if (!descriptor.connector) {
            data.errors.emplace_back(fmt::format("output {} has no connector", descriptor.name));
            return;
        }
        if (!data.output_connector_for_name.try_emplace(descriptor.name, descriptor.connector)
                 .second) {
            data.errors.emplace_back(fmt::format("two outputs are named {}", descriptor.name));
            return;
        }
        data.output_connectors.push_back(descriptor.connector);
        data.output_descriptors[descriptor.connector] = descriptor;
        data.output_connections[descriptor.connector].version_of = version_of;
        data.bind_field_name[descriptor.connector] = "out_" + descriptor.name;
    };
    for (const OutputConnectorDescriptor& descriptor : descriptors) {
        add(descriptor);
    }

    // an in-place input continues as the output of the same name, with the producer's connector
    for (const InputConnectorHandle& input : data.input_connectors) {
        const InputConnectorDescriptor& descriptor = data.input_descriptors.at(input);
        if (!descriptor.in_place) {
            continue;
        }
        const NodeData::PerInputInfo& connected = data.input_connections.at(input);
        const OutputConnectorHandle& source = connected.output;
        if (source && !source->supports_in_place()) {
            data.errors.emplace_back(
                fmt::format("input {} cannot write its source in place", descriptor.name));
            continue;
        }
        const OutputConnectorHandle output = source ? source : std::make_shared<AbsentOutput>();
        if (data.output_connections.contains(output)) {
            data.errors.emplace_back(
                fmt::format("input {} writes a resource in place twice", descriptor.name));
            continue;
        }
        add({.name = descriptor.name,
             .connector = output,
             .access = descriptor.access,
             .disabled = !source,
             .persistent =
                 source && node_data.at(connected.node).output_descriptors.at(source).persistent},
            input);
    }

    for (const auto& [output, input] : data.pass_through_inputs) {
        if (!data.output_connector_for_name.contains(output) &&
            !data.public_outputs.contains(output)) {
            data.errors.emplace_back(
                fmt::format("input {} passes through to {}, which is no output", input, output));
        }
    }
    for (const auto& [output, source] : data.public_outputs) {
        if (data.output_connector_for_name.contains(output)) {
            data.errors.emplace_back(
                fmt::format("declares output {}, which its subgraph provides", output));
        }
    }
    if (!data.errors.empty()) {
        SPDLOG_ERROR("{} failed to describe its outputs and is disabled:\n - {}", data.label,
                     fmt::join(data.errors, "\n - "));
    }
}

bool Graph::validate_in_place_writes(const std::vector<NodeHandle>& nodes) {
    bool valid = true;
    const auto reject = [&](const NodeHandle& node, const std::string& reason) {
        NodeData& data = node_data.at(node);
        SPDLOG_ERROR("{}: {}", data.label, reason);
        data.errors_queued.emplace_back(reason);
        valid = false;
    };
    const auto writes = [&](const NodeHandle& node, const InputConnectorHandle& input) {
        return node_data.at(node).input_descriptors.at(input).in_place;
    };

    // 1. the readers of a version run before its writer, which fails if one depends on it
    std::unordered_map<NodeHandle, std::vector<NodeHandle>> successors;
    for (const auto& [node, predecessors] : compute_predecessors(nodes)) {
        for (const NodeHandle& predecessor : predecessors) {
            successors[predecessor].push_back(node);
        }
    }
    const auto dependents_of = [&](const NodeHandle& writer) {
        std::unordered_set<NodeHandle> dependents;
        for (std::vector<NodeHandle> stack{writer}; !stack.empty();) {
            const auto it = successors.find(stack.back());
            stack.pop_back();
            if (it == successors.end()) {
                continue;
            }
            for (const NodeHandle& successor : it->second) {
                if (dependents.insert(successor).second) {
                    stack.push_back(successor);
                }
            }
        }
        return dependents;
    };
    for (const NodeHandle& writer : nodes) {
        const NodeData& data = node_data.at(writer);
        std::optional<std::unordered_set<NodeHandle>> dependents;
        for (const auto& [output, per_output_info] : data.output_connections) {
            if (!per_output_info.version_of) {
                continue;
            }
            const NodeData::PerInputInfo& source =
                data.input_connections.at(per_output_info.version_of);
            if (!source.node) {
                continue;
            }
            const NodeData& source_data = node_data.at(source.node);
            for (const auto& [reader, reader_input] :
                 source_data.output_connections.at(source.output).inputs) {
                if (!graph_internal::reads_before_writer(
                        node_data.at(reader).input_descriptors.at(reader_input))) {
                    continue;
                }
                if (reader == writer) {
                    reject(writer,
                           fmt::format("cannot write output {} of node {} in place and "
                                       "read it with input {}",
                                       source_data.output_descriptors.at(source.output).name,
                                       source_data.identifier,
                                       data.input_descriptors.at(reader_input).name));
                    break;
                }
                if (!dependents) {
                    dependents = dependents_of(writer);
                }
                if (dependents->contains(reader)) {
                    reject(writer,
                           fmt::format("cannot write output {} of node {} in place after node {} "
                                       "read it, which depends on it",
                                       source_data.output_descriptors.at(source.output).name,
                                       source_data.identifier, node_data.at(reader).identifier));
                    break;
                }
            }
        }
    }
    if (!valid) {
        return false;
    }
    // 2. at most one node writes a version in place
    for (const NodeHandle& node : nodes) {
        const NodeData& data = node_data.at(node);
        for (const auto& [output, per_output_info] : data.output_connections) {
            std::vector<NodeHandle> writers;
            for (const auto& [reader, input] : per_output_info.inputs) {
                if (writes(reader, input)) {
                    writers.push_back(reader);
                }
            }
            if (writers.size() > 1) {
                for (const NodeHandle& writer : writers) {
                    reject(writer,
                           fmt::format("output {} of node {} is written in place by more "
                                       "than one node",
                                       data.output_descriptors.at(output).name, data.identifier));
                }
            }
        }
    }
    if (!valid) {
        return false;
    }

    // 3. delayed readers of a version would see the one written after it
    for (const NodeHandle& node : nodes) {
        const NodeData& data = node_data.at(node);
        for (const auto& [output, per_output_info] : data.output_connections) {
            const auto writer =
                std::ranges::find_if(per_output_info.inputs,
                                     [&](const auto& entry) { return std::apply(writes, entry); });
            if (writer == per_output_info.inputs.end()) {
                continue;
            }
            const auto& [writer_node, writer_input] = *writer;
            for (const auto& [reader, input] : per_output_info.inputs) {
                const InputConnectorDescriptor& descriptor =
                    node_data.at(reader).input_descriptors.at(input);
                if (descriptor.delay > 0) {
                    reject(reader,
                           fmt::format("input {} reads output {} of node {} delayed, which node "
                                       "{} writes in place",
                                       descriptor.name, data.output_descriptors.at(output).name,
                                       data.identifier, node_data.at(writer_node).identifier));
                }
            }
        }
    }
    return valid;
}

std::string Graph::make_error_input_not_connected(const InputConnectorHandle& input,
                                                  const NodeData& data) {
    return fmt::format("the non-optional input {} on {} is not connected.",
                       data.input_descriptors.at(input).name, data.label);
}

} // namespace merian
