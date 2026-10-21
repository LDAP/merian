#include "merian-graph/graph/graph.hpp"

#include "merian-graph/connectors/image/vk_image_out.hpp"
#include "merian-graph/nodes/window/window_node.hpp"
#include "merian/utils/properties_json_dump.hpp"
#include "merian/utils/properties_json_load.hpp"

#include <algorithm>
#include <fmt/ranges.h>
#include <span>
#include <spdlog/spdlog.h>

namespace merian {

void Graph::properties(Properties& props) {
    {
        const BusyScope busy_scope(*this);
        if (props.st_begin_child("graph_properties", "Graph Properties",
                                 Properties::ChildFlagBits::FRAMED)) {
            graph_settings(props);
            props.st_end_child();
        }
        if (props.st_begin_child("profiler", "Profiler Properties",
                                 Properties::ChildFlagBits::FRAMED)) {
            profiler_settings(props);
            props.st_end_child();
        }
    }
    apply_deferred_edits();
}

nlohmann::json Graph::get_properties() {
    JSONDumpProperties props;
    properties(props);
    return props.get();
}

void Graph::set_properties(const nlohmann::json& properties) {
    const std::vector<std::string> problems =
        JSONLoadProperties::load_revealed(properties, [this](Properties& props) {
            this->properties(props);
            return true;
        });
    if (!problems.empty()) {
        throw std::invalid_argument{fmt::format("graph: {}", fmt::join(problems, "; "))};
    }
}

void Graph::graph_settings(Properties& props) {
    if (props.config_uint("iterations in flight", settings.iterations_in_flight, "", 1)) {
        settings.iterations_in_flight = std::max(settings.iterations_in_flight, 1u);
        request_reconnect();
    }
    if (props.config_uint("seed", settings.seed,
                          "Each node derives its random numbers from this seed and its "
                          "identifier, so the same seed renders the same image.")) {
        request_reconnect();
    }
    props.serialize_string("time source", settings.time_source);
    if (props.is_ui()) {
        std::vector<std::string> sources = {TIME_SOURCE_AUTO, TIME_SOURCE_SYSTEM, TIME_SOURCE_TIME,
                                            TIME_SOURCE_DELTA};
        for (const NodeHandle& provider : time_providers()) {
            sources.emplace_back(node_data.at(provider).identifier);
        }
        // a selected source that is currently not providing stays visible
        if (std::find(sources.begin(), sources.end(), settings.time_source) == sources.end()) {
            sources.emplace_back(settings.time_source);
        }
        int selected = static_cast<int>(std::distance(
            sources.begin(), std::find(sources.begin(), sources.end(), settings.time_source)));
        if (props.config_options("time source", selected, sources, Properties::OptionsStyle::COMBO,
                                 "Which node or overwrite drives the graph clock. Auto picks the "
                                 "first time provider in topological order.")) {
            settings.time_source = sources[selected];
        }
    }
    if (settings.time_source == TIME_SOURCE_TIME) {
        float time_s = to_seconds(run_info.elapsed);
        if (props.config_float("time", time_s, "The graph time in seconds.", 0.1)) {
            settings.time_overwrite_delta_ms += (time_s - to_seconds(run_info.elapsed)) * 1000.;
        }
        float offset_s = 0;
        if (props.config_float("offset", offset_s, "Seconds added to the graph time.", 0.01)) {
            settings.time_overwrite_delta_ms += offset_s * 1000.;
        }
    } else if (settings.time_source == TIME_SOURCE_DELTA) {
        props.config_float("delta", settings.time_overwrite_delta_ms,
                           "Milliseconds the graph time advances per iteration.", 0.001);
        float fps = 1000. / settings.time_overwrite_delta_ms;
        if (props.config_float("fps", fps, "", 0.01)) {
            settings.time_overwrite_delta_ms = 1000 / fps;
        }
    }

    props.st_separate();
    if (props.config_bool("fps limiter", settings.limit_fps) && settings.limit_fps != 0) {
        settings.limit_fps = 60;
    }
    if (settings.limit_fps != 0) {
        if (props.config_int("fps limit", settings.limit_fps, "")) {
            settings.limit_fps = std::max(1, settings.limit_fps);
        }
    }

    props.st_separate();
    if (props.config_bool("alias transient resources", settings.alias_transient_resources,
                          "Outputs that are not persistent share memory where their lifetimes do "
                          "not overlap.")) {
        request_reconnect();
    }
    if (props.config_uint("max async tasks", settings.max_async_tasks,
                          "Async host callbacks, e.g. image writes, that run alongside the graph; "
                          "it waits while this many run and another one waits.",
                          1)) {
        settings.max_async_tasks = std::max(settings.max_async_tasks, 1u);
        async_tasks.reset();
    }

    props.st_separate();
    if (props.config_text("imgui event", settings.imgui_event, true,
                          "Comma-separated events from ImGui nodes that draw this graph's UI. "
                          "Press enter to confirm.")) {
        request_reconnect();
    }
}

void Graph::profiler_settings(Properties& props) {
    props.config_uint("report interval", settings.profiler_report_interval_ms,
                      "Set the time period for the profiler to update in ms. Meaning, averages and "
                      "deviations are calculated over this period.");
    props.config_uint("evict after", settings.profiler_evict_after_ms,
                      "Drop entries that have not run for this many milliseconds. Prevents the "
                      "section tree from growing indefinitely while keeping briefly-silent "
                      "sections visible to avoid layout flicker.");
}

void Graph::node_properties(const std::string& identifier, Properties& props) {
    {
        const BusyScope busy_scope(*this);
        node_section(props, node_for(identifier), false);
    }
    apply_deferred_edits();
}

void Graph::node_section(Properties& props, const NodeHandle& node, const bool edit) {
    NodeData& data = node_data.at(node);
    const std::string node_label = fmt::format("[{}] {}", data.state_name(), data.label);
    if (!props.st_begin_child(data.identifier, node_label)) {
        return;
    }

    const auto show_in_window = [&](const std::string& output) {
        // children are connected through their parent
        if (!edit || data.parent) {
            return;
        }
        const ConnectorRef carried = carrier({node, output});
        const NodeData& carried_data = node_data.at(carried.node);
        const auto connector = carried_data.output_connector_for_name.find(carried.name);
        if (connector == carried_data.output_connector_for_name.end()) {
            return;
        }
        const OutputConnectorDescriptor& descriptor =
            carried_data.output_descriptors.at(connector->second);
        if (!descriptor.scratch && !descriptor.disabled &&
            std::dynamic_pointer_cast<VkImageOut>(connector->second) &&
            props.config_bool("Show in Window")) {
            const std::string window =
                add_node(registry.node_type_info<WindowNode>().node_type_name);
            add_connection(data.identifier, window, output, std::string{WindowNode::DISPLAY_INPUT});
        }
    };
    const auto remove_connection_control = [&](const std::string& input) {
        if (!edit) {
            return;
        }
        const auto it = data.desired_incoming_connections.find(input);
        if (it != data.desired_incoming_connections.end() &&
            props.config_bool("Remove Connection")) {
            remove_connection(it->second.node, node, input);
        }
    };

    // the parent owns its children and their settings
    if (!data.parent) {
        if (props.config_bool("enabled", data.enabled)) {
            request_reconnect();
        }
        if (edit) {
            props.st_no_space();
            if (props.config_bool("Remove")) {
                remove_node(data.identifier);
            }
        }
        if (props.config_uint("linearization order", data.linearization_order,
                              "Within-layer execution order: lower runs earlier, ties by "
                              "identifier. Cannot violate dependencies.")) {
            request_reconnect();
        }
    }

    if (!data.errors.empty()) {
        props.output_text(fmt::format("Errors:\n  - {}", fmt::join(data.errors, "\n  - ")));
    }
    if (data.unsupported) {
        props.output_text("Unsupported: {}", data.unsupported_reason);
    }
    if (data.force_disabled) {
        props.output_text("Force disabled: {}", data.force_disabled_reason);
    }

    props.st_separate();
    if (props.st_begin_child("sections", "", Properties::ChildFlagBits::TABS)) {
        if (!data.parent && props.st_begin_child("properties", "Properties",
                                                 Properties::ChildFlagBits::DEFAULT_OPEN)) {
            apply_properties_status(data.identifier, node->properties(props));
            props.st_end_child();
        }
        if (!needs_reconnect &&
            (!data.output_connections.empty() || !data.public_outputs.empty()) &&
            props.st_begin_child("outputs", "Outputs")) {
            for (const auto& [output, per_output_info] : data.output_connections) {
                const OutputConnectorDescriptor& descriptor = data.output_descriptors.at(output);
                if (props.st_begin_child(
                        descriptor.name,
                        fmt::format("{}{}{}{}", descriptor.name,
                                    descriptor.scratch ? " (scratch)" : "",
                                    per_output_info.version_of ? " (in place)" : "",
                                    descriptor.disabled ? " (disabled)" : ""),
                        Properties::ChildFlagBits::DEFAULT_OPEN)) {
                    show_in_window(descriptor.name);
                    output_section(props, data, output);
                    props.st_end_child();
                }
            }
            const std::map<std::string, ConnectorRef> public_outputs(data.public_outputs.begin(),
                                                                     data.public_outputs.end());
            for (const auto& [name, source] : public_outputs) {
                const std::string label = fmt::format("{} (public)", name);
                if (props.st_begin_child(label, label, Properties::ChildFlagBits::DEFAULT_OPEN)) {
                    show_in_window(name);
                    public_output_section(props, source);
                    props.st_end_child();
                }
            }
            props.st_end_child();
        }
        if (!needs_reconnect && (!data.input_connectors.empty() || !data.public_inputs.empty()) &&
            props.st_begin_child("inputs", "Inputs")) {
            for (const InputConnectorHandle& input : data.input_connectors) {
                const std::string& name = data.input_descriptors.at(input).name;
                if (props.st_begin_child(name, name)) {
                    input_section(props, data, input);
                    remove_connection_control(name);
                    props.st_end_child();
                }
            }
            const std::map<std::string, std::vector<ConnectorRef>> public_inputs(
                data.public_inputs.begin(), data.public_inputs.end());
            for (const auto& [name, fed_inputs] : public_inputs) {
                const std::string label = fmt::format("{} (public)", name);
                if (props.st_begin_child(label, label)) {
                    public_input_section(props, node, name, fed_inputs);
                    remove_connection_control(name);
                    props.st_end_child();
                }
            }
            props.st_end_child();
        }
        if (!data.children.empty() && props.st_begin_child("subnodes", "Subnodes")) {
            for (const NodeHandle& child : data.children) {
                node_section(props, child, edit);
            }
            props.st_end_child();
        }
        props.st_end_child();
    }
    props.st_end_child();
}

void Graph::output_section(Properties& props, NodeData& data, const OutputConnectorHandle& output) {
    const auto& per_output_info = data.output_connections.at(output);
    std::vector<std::string> receivers;
    receivers.reserve(per_output_info.inputs.size());
    for (const auto& [node, input] : per_output_info.inputs) {
        const std::string& input_name = node_data.at(node).input_descriptors.at(input).name;
        receivers.emplace_back(fmt::format("({}, {})", input_name, node_data.at(node).label));
    }

    const std::span<const ResourceData::Copy> copies =
        per_output_info.resource_data
            ? std::span<const ResourceData::Copy>(per_output_info.resource_data->copies)
            : std::span<const ResourceData::Copy>();
    std::string current_copy_index = "none";
    if (!copies.empty() && !per_output_info.precomputed_resources.empty()) {
        current_copy_index = fmt::format("{:02}", data.set_index(run_iteration) % copies.size());
    }

    props.output_text(fmt::format("# Copies: {:02}\nCopy index: {}\nSending to: [{}]",
                                  copies.size(), current_copy_index, fmt::join(receivers, ", ")));

    if (props.st_begin_child("connector", "Connector Properties")) {
        output->properties(props);
        props.st_end_child();
    }
    if (props.st_begin_child("resources", "Resource Properties")) {
        for (uint32_t i = 0; i < copies.size(); i++) {
            if (props.st_begin_child(fmt::format("resource_{}", i), fmt::format("Copy {:02}", i))) {
                copies[i].resource->properties(props);
                props.st_end_child();
            }
        }
        props.st_end_child();
    }
}

void Graph::public_output_section(Properties& props, const ConnectorRef& source) {
    const NodeData& source_data = node_data.at(source.node);
    std::vector<std::string> receivers;
    if (const auto connector = source_data.output_connector_for_name.find(source.name);
        connector != source_data.output_connector_for_name.end()) {
        if (const auto connections = source_data.output_connections.find(connector->second);
            connections != source_data.output_connections.end()) {
            for (const auto& [reader, input] : connections->second.inputs) {
                receivers.emplace_back(
                    fmt::format("({}, {})", node_data.at(reader).input_descriptors.at(input).name,
                                node_data.at(reader).label));
            }
        }
    }
    props.output_text("Carried by: {}.{}\nSending to: [{}]", source_data.identifier, source.name,
                      fmt::join(receivers, ", "));
}

void Graph::input_section(Properties& props, NodeData& data, const InputConnectorHandle& input) {
    props.st_separate("Input Properties");
    input->properties(props);
    props.st_separate("Connection");
    if (!data.input_connections.contains(input)) {
        props.output_text("Input not connected.");
        return;
    }
    const auto& per_input_info = data.input_connections.at(input);
    if (!per_input_info.output) {
        props.output_text("Optional input not connected.");
        return;
    }
    const NodeData& source_data = node_data.at(per_input_info.node);
    props.output_text(fmt::format("Receiving from: {}, {}",
                                  source_data.output_descriptors.at(per_input_info.output).name,
                                  source_data.label));
}

void Graph::public_input_section(Properties& props,
                                 const NodeHandle& node,
                                 const std::string& input,
                                 const std::vector<ConnectorRef>& fed_inputs) {
    std::vector<std::string> fed;
    for (const auto& [child, child_input] : fed_inputs) {
        const NodeData& child_data = node_data.at(child);
        fed.emplace_back(fmt::format(
            "{}.{}{}", child_data.identifier, child_input,
            child_data.public_feeds.contains(child_input) ? "" : " (its auto connection)"));
    }
    props.output_text("Feeds: [{}]", fmt::join(fed, ", "));
    if (const std::optional<Connection> wired = wiring(node, input)) {
        props.output_text("Receiving from: {}, {}", wired->src_output,
                          node_data.at(wired->src).label);
    } else {
        props.output_text("Input not connected.");
    }
}

nlohmann::json Graph::node_get_properties(const std::string& identifier) const {
    return node_for(identifier)->dump_config();
}

void Graph::node_set_properties(const std::string& identifier, const nlohmann::json& properties) {
    if (const auto problems = load_node_properties(identifier, node_for(identifier), properties)) {
        throw std::invalid_argument{*problems};
    }
}

std::optional<std::string> Graph::load_node_properties(const std::string& identifier,
                                                       const NodeHandle& node,
                                                       const nlohmann::json& properties) {
    std::vector<std::string> problems;
    apply_properties_status(identifier, node->load_config(properties, problems));
    if (problems.empty()) {
        return std::nullopt;
    }
    // the node may have removed itself
    return fmt::format("node '{}' ({}): {}", identifier, registry.node_type_name(node),
                       fmt::join(problems, "; "));
}

void Graph::apply_properties_status(const std::string& identifier,
                                    const Node::NodeStatusFlags flags) {
    if ((flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT) != 0u) {
        SPDLOG_DEBUG("node {} requested reconnect", identifier);
        request_reconnect();
    }
    if ((flags & Node::NodeStatusFlagBits::REMOVE_NODE) != 0u) {
        remove_node(identifier);
    }
}

} // namespace merian
