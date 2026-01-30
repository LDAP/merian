#include "merian-nodes/graph/graph.hpp"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace merian {

void Graph::properties(Properties& props) {
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
        props.output_text("Iterations in flight {:02}/{:02}", ring_fences.count_waiting(),
                          ring_fences.size());

        props.st_separate();
        if (props.config_uint("iterations in flight", desired_iterations_in_flight)) {
            request_reconnect();
        }
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
        props.output_text("{}: {}", registry.node_names()[new_node_selected],
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
        if (autodetect_dst_input && add_connection_selected_src_output < (int)src_outputs.size()) {
            // maybe there is a input that is named exactly like the output
            for (int i = 0; i < static_cast<int>(dst_inputs.size()); i++) {
                if (dst_inputs[i] == src_outputs[add_connection_selected_src_output]) {
                    add_connection_selected_dst_input = i;
                }
            }
        }
        props.config_options("connection dst input", add_connection_selected_dst_input, dst_inputs,
                             Properties::OptionsStyle::COMBO);
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
                                  registry.node_type_name(it->second.first));
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
                props.st_begin_child("run", "Graph Run", Properties::ChildFlagBits::DEFAULT_OPEN)) {
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
                                         registry.node_type_name(node));
            }

            if (props.st_begin_child(identifier, node_label)) {
                NodeHandle node;
                std::string type;

                // Create Node
                if (!loading) {
                    node = node_for_identifier.at(identifier);
                    type = registry.node_type_name(node);
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
                    if ((flags & Node::NodeStatusFlagBits::REMOVE_NODE) != 0u) {
                        remove_node(data.identifier);
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
                add_connection(j_con["src"], j_con["dst"], j_con["src_output"], j_con["dst_input"]);
            }
        }
    }
}

void Graph::io_props_for_node(Properties& config, NodeHandle& node, NodeData& data) {
    if (data.descriptor_set_layout &&
        config.st_begin_child("desc_set_layout", "Descriptor Set Layout")) {
        config.output_text(fmt::format("{}", data.descriptor_set_layout));
        config.st_end_child();
    }
    if (!needs_reconnect && !data.output_connections.empty() &&
        config.st_begin_child("outputs", "Outputs")) {
        for (auto& [output, per_output_info] : data.output_connections) {
            if (config.st_begin_child(output->name, output->name)) {
                std::vector<std::string> receivers;
                receivers.reserve(per_output_info.inputs.size());
                for (auto& [node, input] : per_output_info.inputs) {
                    receivers.emplace_back(fmt::format("({}, {} ({}))", input->name,
                                                       node_data.at(node).identifier,
                                                       registry.node_type_name(node)));
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
                    per_output_info.descriptor_set_binding == DescriptorSet::NO_DESCRIPTOR_BINDING
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
    if (!needs_reconnect && !data.input_connectors.empty() &&
        config.st_begin_child("inputs", "Inputs")) {
        for (const auto& input : data.input_connectors) {
            if (config.st_begin_child(input->name, input->name)) {
                config.st_separate("Input Properties");
                input->properties(config);
                config.st_separate("Connection");
                if (data.input_connections.contains(input)) {
                    auto& per_input_info = data.input_connections.at(input);
                    config.output_text(
                        fmt::format("Descriptor set binding: {}",
                                    per_input_info.descriptor_set_binding ==
                                            DescriptorSet::NO_DESCRIPTOR_BINDING
                                        ? "None"
                                        : std::to_string(per_input_info.descriptor_set_binding)));
                    if (per_input_info.output) {
                        config.output_text(
                            fmt::format("Receiving from: {}, {} ({})", per_input_info.output->name,
                                        node_data.at(per_input_info.node).identifier,
                                        registry.node_type_name(per_input_info.node)));
                    } else {
                        config.output_text("Optional input not connected.");
                    }
                } else {
                    config.output_text("Input not connected.");
                }

                if (data.desired_incoming_connections.contains(input->name) &&
                    config.config_bool("Remove Connection")) {
                    auto& incoming_conection = data.desired_incoming_connections.at(input->name);
                    remove_connection(incoming_conection.first, node, input->name);
                }

                config.st_end_child();
            }
        }
        config.st_end_child();
    }
}

} // namespace merian
