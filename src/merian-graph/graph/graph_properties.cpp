#include "merian-graph/graph/graph.hpp"

#include "merian-graph/connectors/image/vk_image_in.hpp"
#include "merian-graph/nodes/window/window_node.hpp"

#include <algorithm>
#include <span>
#include <spdlog/spdlog.h>

namespace merian {

namespace {

void node_memory_properties(Properties& props,
                            const GraphMemoryReport& report,
                            const std::string& identifier) {
    props.output_text(report.format_node(identifier));
}

} // namespace

void Graph::graph_properties(Properties& props) {
    props.output_text("Run iteration: {}", run_iteration);
    props.output_text("Run Elapsed: {:%H:%M:%S}s", duration_elapsed_since_connect);
    props.output_text("Total iterations: {}", total_iteration);
    props.output_text("Total Elapsed: {:%H:%M:%S}s", duration_elapsed);
    props.output_text("Time delta: {:04f}ms", to_milliseconds(time_delta));
    props.output_text("GPU wait: {:04f}ms", to_milliseconds(gpu_wait_time));
    props.output_text("Iterations in flight {:02}/{:02}", ring_fences.count_waiting(),
                      ring_fences.size());

    props.st_separate();
    if (props.config_uint("iterations in flight", desired_iterations_in_flight)) {
        request_reconnect();
    }
    props.serialize_string("time source", time_source);
    if (props.is_ui()) {
        std::vector<std::string> sources = {TIME_SOURCE_AUTO, TIME_SOURCE_SYSTEM, TIME_SOURCE_TIME,
                                            TIME_SOURCE_DELTA};
        for (const NodeHandle& provider : time_providers()) {
            sources.emplace_back(node_data.at(provider).identifier);
        }
        // a selected source that is currently not providing stays visible
        if (std::find(sources.begin(), sources.end(), time_source) == sources.end()) {
            sources.emplace_back(time_source);
        }
        int selected = static_cast<int>(
            std::distance(sources.begin(), std::find(sources.begin(), sources.end(), time_source)));
        if (props.config_options("time source", selected, sources, Properties::OptionsStyle::COMBO,
                                 "Which node or overwrite drives the graph clock. Auto picks the "
                                 "first time provider in topological order.")) {
            time_source = sources[selected];
        }
    }
    if (time_source == TIME_SOURCE_TIME) {
        float time_s = to_seconds(duration_elapsed);
        props.config_float("time (s)", time_s, "", 0.1);
        float delta_s = time_s - to_seconds(duration_elapsed);
        props.config_float("offset (s)", delta_s, "", 0.01);
        time_overwrite_delta_ms += delta_s * 1000.;
    } else if (time_source == TIME_SOURCE_DELTA) {
        props.config_float("delta (ms)", time_overwrite_delta_ms, "", 0.001);
        float fps = 1000. / time_overwrite_delta_ms;
        props.config_float("fps", fps, "", 0.01);
        time_overwrite_delta_ms = 1000 / fps;
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
    if (limit_fps > 0) {
        const InFlightData& in_flight_data = ring_fences.get().user_data;
        props.output_text("CPU sleep time: {:04f}ms",
                          to_milliseconds(in_flight_data.cpu_sleep_time));
    }

    props.st_separate();
    if (props.is_ui() && !layers.empty()) {
        props.output_text("dependency layers: {}", layers.size());
    }
    if (props.config_bool("alias transient resources", alias_transient_resources,
                          "Outputs that are not persistent share memory where their lifetimes do "
                          "not overlap.")) {
        request_reconnect();
    }

    props.st_separate();
    props.config_bool("flush thread pool", flush_thread_pool_at_run_start,
                      "If enabled, the tasks queue of the thread pool is flushed when a "
                      "run starts. HIGHLY RECOMMENDED as it limits memory allocations and "
                      "prevents the queue to fill up indefinitely.");
    props.output_text("tasks in queue: {}", thread_pool->queue_size());

    props.st_separate();
    static_cast<void>(
        props.config_text("imgui event", imgui_event, false,
                          "Comma-separated events from ImGui nodes that draw this graph's UI."));

    if (props.is_ui()) {
        props.st_separate("Events");
        const auto now = std::chrono::steady_clock::now();
        std::erase_if(recent_events, [&](const auto& entry) {
            return now - entry.second.time > std::chrono::seconds(3);
        });
        std::string list;
        for (const auto& [name, event] : recent_events) {
            if (!list.empty())
                list += ", ";
            list += fmt::format("{} (it {})", name, event.iteration);
        }
        props.output_text("Last events: {}", list.empty() ? "<none>" : list);
    }
}

void Graph::profiler_properties(Properties& props) {
#ifdef MERIAN_PROFILER_ENABLE
    // a runtime switch, not stored
    if (props.is_ui()) {
        props.config_bool("profiling", profiler_enable);
    }
#else
    profiler_enable = false;
    props.output_text("Profiler disabled at compile-time!\n\n Enable with 'meson configure "
                      "<builddir> -Dmerian:performance_profiling=true'.");
#endif

    if (profiler_enable) {
        props.st_no_space();
        props.config_uint("report intervall", profiler_report_intervall_ms,
                          "Set the time period for the profiler to update in ms. Meaning, "
                          "averages and deviations are calculated over this period.");
        props.config_uint("evict after ms", profiler_evict_after_ms,
                          "Drop entries that have not run for this many milliseconds. "
                          "Prevents the section tree from growing indefinitely while keeping "
                          "briefly-silent sections visible to avoid layout flicker.");

        if (last_run_report &&
            props.st_begin_child("run", "Graph Run", Properties::ChildFlagBits::DEFAULT_OPEN)) {
            if (!last_run_report.cpu_report.empty()) {
                props.st_separate("CPU");
                const float* cpu_samples = &cpu_time_history[time_history_current + 1];
                if (cpu_auto) {
                    cpu_max =
                        *std::max_element(cpu_samples, cpu_samples + cpu_time_history.size() - 1);
                }

                props.output_plot_line("cpu", cpu_samples, cpu_time_history.size() - 1, 0, cpu_max);
                cpu_auto &= !props.config_float("cpu max ms", cpu_max, "", 0.1f, 0.0f, 1000.0f);
                props.st_no_space();
                props.config_bool("cpu auto", cpu_auto);
                Profiler::get_cpu_report_as_config(props, last_run_report);
            }

            if (!last_run_report.gpu_report.empty()) {
                props.st_separate("GPU");
                const float* gpu_samples = &gpu_time_history[time_history_current + 1];
                if (gpu_auto) {
                    gpu_max =
                        *std::max_element(gpu_samples, gpu_samples + gpu_time_history.size() - 1);
                }

                props.output_plot_line("gpu", gpu_samples, gpu_time_history.size() - 1, 0, gpu_max);
                gpu_auto &= !props.config_float("gpu max ms", gpu_max, "", 0.1f, 0.0f, 1000.0f);
                props.st_no_space();
                props.config_bool("gpu auto", gpu_auto);
                Profiler::get_gpu_report_as_config(props, last_run_report);
            }
            props.st_end_child();
        }
    }
    if (last_build_report && props.st_begin_child("build", "Last Graph Build")) {
        Profiler::get_report_as_config(props, last_build_report);
        props.st_end_child();
    }
}

void Graph::properties(Properties& props) {
    {
        const BusyScope busy_scope(*this);
        draw_properties(props);
    }
    apply_deferred_edits();
}

void Graph::draw_properties(Properties& props) {
    needs_reconnect |= props.config_bool("Rebuild");
    props.st_no_space();
    props.output_text("Run iteration: {}", run_iteration);
    if (props.config_bool("reload shaders",
                          "Recompile shaders whose source files changed on disk.")) {
        send_graph_event("reload_shaders", std::any(false));
    }
    props.st_no_space();
    if (props.config_bool("reload shaders (force)",
                          "Recompile all shaders unconditionally, ignoring mtimes and caches.")) {
        send_graph_event("reload_shaders", std::any(true));
    }

    if (props.is_ui()) {
        props.st_separate("Load / Store");
        static_cast<void>(props.config_text("path", store_path, false, "Path for Load and Store."));
        if (props.config_bool("Store") && !store_path.empty()) {
            store_to_file(store_path);
        }
        props.st_no_space();
        if (props.config_bool("Load") && !store_path.empty()) {
            when_idle([this, path = store_path] {
                try {
                    load_from_file(path);
                } catch (const std::exception& e) {
                    SPDLOG_ERROR("cannot load {}: {}", path, e.what());
                }
            });
        }
    }

    props.st_separate("Events");
    if (props.is_ui() && props.config_text("send event", props_send_event, true) &&
        !props_send_event.empty()) {
        send_event(props_send_event);
        props_send_event.clear();
    }
    if (props.st_begin_child("graph_properties", "Graph Properties",
                             Properties::ChildFlagBits::FRAMED)) {
        graph_properties(props);
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
            try {
                add_node(registry.node_names()[new_node_selected], optional_identifier);
            } catch (const std::invalid_argument& e) {
                SPDLOG_ERROR("cannot add node: {}", e.what());
            }
        }
        props.output_text("{}: {}", registry.node_names()[new_node_selected],
                          registry.node_info(registry.node_names()[new_node_selected]).description);

        // children are edited through their parent
        std::vector<std::string> node_ids;
        std::ranges::transform(
            top_level_nodes(), std::back_inserter(node_ids),
            [&](const NodeHandle& node) { return node_data.at(node).identifier; });
        if (!node_ids.empty()) {
            const int last_node = static_cast<int>(node_ids.size()) - 1;
            add_connection_selected_src = std::min(add_connection_selected_src, last_node);
            add_connection_selected_dst = std::min(add_connection_selected_dst, last_node);
            remove_node_selected = std::min(remove_node_selected, last_node);
            props.st_separate("Add Connection");
            bool autodetect_dst_input = false;
            if (props.config_options("connection src", add_connection_selected_src, node_ids,
                                     Properties::OptionsStyle::COMBO)) {
                add_connection_selected_src_output = 0;
                autodetect_dst_input = true;
            }
            std::vector<std::string> src_outputs;
            const NodeData& src_data =
                node_data.at(node_for_identifier.at(node_ids[add_connection_selected_src]));
            for (const auto& [output_name, output] : src_data.output_connector_for_name) {
                if (!src_data.output_descriptors.at(output).scratch) {
                    src_outputs.emplace_back(output_name);
                }
            }
            for (const auto& [port, source] : src_data.public_outputs) {
                src_outputs.emplace_back(port);
            }
            std::sort(src_outputs.begin(), src_outputs.end());
            autodetect_dst_input |=
                props.config_options("connection src output", add_connection_selected_src_output,
                                     src_outputs, Properties::OptionsStyle::COMBO);
            if (props.config_options("connection dst", add_connection_selected_dst, node_ids,
                                     Properties::OptionsStyle::COMBO)) {
                add_connection_selected_dst_input = 0;
                autodetect_dst_input = true;
            }
            const NodeData& dst_data =
                node_data.at(node_for_identifier.at(node_ids[add_connection_selected_dst]));
            std::vector<std::string> dst_inputs;
            dst_inputs.reserve(dst_data.input_connector_for_name.size());
            for (const auto& [input_name, input] : dst_data.input_connector_for_name) {
                dst_inputs.emplace_back(input_name);
            }
            for (const auto& [port, inputs] : dst_data.public_inputs) {
                if (!dst_data.input_connector_for_name.contains(port)) {
                    dst_inputs.emplace_back(port);
                }
            }
            std::sort(dst_inputs.begin(), dst_inputs.end());
            if (autodetect_dst_input &&
                add_connection_selected_src_output < static_cast<int>(src_outputs.size())) {
                // an input named like the output
                for (int i = 0; i < static_cast<int>(dst_inputs.size()); i++) {
                    if (dst_inputs[i] == src_outputs[add_connection_selected_src_output]) {
                        add_connection_selected_dst_input = i;
                    }
                }
            }
            props.config_options("connection dst input", add_connection_selected_dst_input,
                                 dst_inputs, Properties::OptionsStyle::COMBO);
            const bool valid_connection =
                add_connection_selected_src_output < static_cast<int>(src_outputs.size()) &&
                add_connection_selected_dst_input < static_cast<int>(dst_inputs.size());
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
                    props.output_text("Warning: Input already connected with {}, {}",
                                      it->second.name, node_data.at(it->second.node).label);
                }
            }
            props.st_separate("Remove Node");
            props.config_options("remove identifier", remove_node_selected, node_ids,
                                 Properties::OptionsStyle::COMBO);
            if (props.config_bool("Remove Node")) {
                remove_node(node_ids[remove_node_selected]);
            }
        }

        props.st_end_child();
    }

    if (props.st_begin_child("profiler", "Profiler", Properties::ChildFlagBits::FRAMED)) {
        profiler_properties(props);
        props.st_end_child();
    }

    if (props.st_begin_child("memory", "Memory", Properties::ChildFlagBits::FRAMED)) {
        memory_properties(props);
        props.st_end_child();
    }

    if (props.st_begin_child("nodes", "Nodes",
                             Properties::ChildFlagBits::DEFAULT_OPEN |
                                 Properties::ChildFlagBits::FRAMED)) {
        for (const NodeHandle& node : top_level_nodes()) {
            node_properties(props, node);
        }
        props.st_end_child();
    }
}

void Graph::node_properties(Properties& props, const NodeHandle& node) {
    NodeData& data = node_data.at(node);
    const std::string node_label = fmt::format("[{}] {}", data.state_name(), data.label);
    if (!props.st_begin_child(data.identifier, node_label)) {
        return;
    }

    // the parent owns its children and their settings
    if (!data.parent) {
        if (props.config_bool("enabled", data.enabled))
            request_reconnect();
        props.st_no_space();
        if (props.config_bool("Remove")) {
            remove_node(data.identifier);
        }
        if (props.config_uint("linearization order", data.linearization_order,
                              "Within-layer execution order: lower runs earlier, ties by "
                              "identifier. Cannot violate dependencies.")) {
            request_reconnect();
        }
    }

    if (!data.errors.empty()) {
        props.output_text(fmt::format("Errors:\n  - {}", fmt::join(data.errors, "\n   - ")));
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
        io_props_for_node(props, node, data);
        if (props.is_ui() && props.st_begin_child("memory", "Memory")) {
            node_memory_properties(props, memory_view_for_ui().report, data.identifier);
            props.st_end_child();
        }
        if (!data.children.empty() && props.st_begin_child("subnodes", "Subnodes")) {
            for (const NodeHandle& child : data.children) {
                node_properties(props, child);
            }
            props.st_end_child();
        }
        props.st_end_child();
    }
    props.st_end_child();
}

GraphMemoryReport Graph::get_memory_report() const {
    GraphMemoryReport report;
    report.layer_count = static_cast<uint32_t>(layers.size());
    report.period = period;
    report.aliasing = alias_transient_resources;

    // in run order, then the nodes that do not run
    std::vector<NodeHandle> nodes;
    if (!needs_reconnect) {
        for (const Layer& layer : layers) {
            nodes.insert(nodes.end(), layer.nodes.begin(), layer.nodes.end());
        }
    }
    const std::unordered_set<NodeHandle> running(nodes.begin(), nodes.end());
    for (const auto& [identifier, node] : node_for_identifier) {
        if (!running.contains(node)) {
            nodes.push_back(node);
        }
    }

    for (const NodeHandle& node : nodes) {
        const NodeData& data = node_data.at(node);
        const std::string& identifier = data.identifier;
        report.node_allocations.emplace_back(identifier, data.label,
                                             data.memory_tracker->get_allocated_size());

        // a pending reconnect may have left the outputs half allocated
        if (needs_reconnect) {
            continue;
        }
        for (const auto& [output, per_output_info] : data.output_connections) {
            // versions written in place share the resources of their root
            if (!per_output_info.resource_data || per_output_info.version_of) {
                continue;
            }
            const std::vector<ResourceData::Copy>& copies = per_output_info.resource_data->copies;
            const OutputConnectorDescriptor& descriptor = data.output_descriptors.at(output);
            for (uint32_t copy_index = 0; copy_index < copies.size(); copy_index++) {
                const TransientMemoryScopeHandle& scope = copies[copy_index].scope;
                const std::string name =
                    copies.size() == 1
                        ? fmt::format("{}.{}", identifier, descriptor.name)
                        : fmt::format("{}.{} ({})", identifier, descriptor.name, copy_index);
                for (const TransientMemoryAllocator::Placement& placement :
                     scope->get_placements()) {
                    report.placed.emplace_back(identifier, name, descriptor.scratch, placement,
                                               scope->get_lifetime());
                }
                for (const TransientMemoryAllocator::Fallback& fallback : scope->get_fallbacks()) {
                    report.unplaced.emplace_back(identifier, name, descriptor.scratch, fallback);
                }
            }
        }
    }
    if (!needs_reconnect) {
        for (const TransientMemoryAllocator::Block& block : transient_blocks) {
            report.blocks.emplace_back(block.memory_type_index, block.size, block.requested_size,
                                       block.memory);
        }
        for (const Layer& layer : layers) {
            std::vector<std::string>& labels = report.layers.emplace_back();
            for (const NodeHandle& node : layer.nodes) {
                labels.push_back(node_data.at(node).identifier);
            }
        }
        report.in_use.assign(static_cast<size_t>(period) * layers.size(), 0);
        for (const GraphMemoryReport::Placed& placed : report.placed) {
            for (uint32_t position = 0; position < report.in_use.size(); position++) {
                if (placed.lifetime.contains(position)) {
                    report.in_use[position] += placed.placement.size;
                }
            }
        }
    }

    std::ranges::sort(report.placed, [](const auto& a, const auto& b) {
        return std::tie(a.placement.block_index, a.placement.offset, a.placement.size, a.output) <
               std::tie(b.placement.block_index, b.placement.offset, b.placement.size, b.output);
    });
    std::ranges::sort(report.unplaced, [](const auto& a, const auto& b) {
        return std::tie(b.fallback.size, a.output) < std::tie(a.fallback.size, b.output);
    });

    return report;
}

const Graph::MemoryView& Graph::memory_view_for_ui() {
    const auto now = std::chrono::steady_clock::now();
    if (memory_view_dirty || now - memory_view_time > std::chrono::seconds(1)) {
        memory_view_dirty = false;
        memory_view_time = now;
        memory_view.report = get_memory_report();
        memory_view.summary = memory_view.report.format_summary();
        memory_view.in_use = memory_view.report.format_in_use();
        memory_view.blocks = memory_view.report.format_blocks();
    }
    return memory_view;
}

void Graph::memory_properties(Properties& props) {
    if (!props.is_ui()) {
        return;
    }

    const MemoryView& view = memory_view_for_ui();
    const GraphMemoryReport& report = view.report;
    if (!report.in_use.empty()) {
        std::vector<float> in_use_mib;
        in_use_mib.reserve(report.in_use.size());
        for (const vk::DeviceSize size : report.in_use) {
            in_use_mib.push_back(static_cast<float>(size) / (1024 * 1024));
        }
        props.output_plot_line("in use (MiB) per layer over the period", in_use_mib.data(),
                               static_cast<uint32_t>(in_use_mib.size()), 0);
    }
    props.output_text(view.summary);
    if (!view.in_use.empty() && props.st_begin_child("in_use", "In Use")) {
        props.output_text(view.in_use);
        props.st_end_child();
    }
    if (!view.blocks.empty() && props.st_begin_child("blocks", "Blocks")) {
        props.output_text(view.blocks);
        props.st_end_child();
    }
    if (props.st_begin_child("nodes", "Nodes", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        for (const GraphMemoryReport::NodeAllocation& node : report.node_allocations) {
            if (props.st_begin_child(node.node,
                                     fmt::format("{}: {} allocated, {} placed", node.label,
                                                 format_size(node.size),
                                                 format_size(report.placed_size(node.node))))) {
                node_memory_properties(props, report, node.node);
                props.st_end_child();
            }
        }
        props.st_end_child();
    }
}

// Probes which outputs a WindowNode could display. can_receive_from is side-effect free, so one
// instance serves all queries.
static const VkImageInHandle window_display_input = VkImageIn::create();

void Graph::io_props_for_node(Properties& config, const NodeHandle& node, NodeData& data) {
    if (!needs_reconnect && (!data.output_connections.empty() || !data.public_outputs.empty()) &&
        config.st_begin_child("outputs", "Outputs")) {
        for (auto& [output, per_output_info] : data.output_connections) {
            const OutputConnectorDescriptor& descriptor = data.output_descriptors.at(output);

            if (config.st_begin_child(descriptor.name,
                                      fmt::format("{}{}{}{}", descriptor.name,
                                                  descriptor.scratch ? " (scratch)" : "",
                                                  per_output_info.version_of ? " (in place)" : "",
                                                  descriptor.disabled ? " (disabled)" : ""),
                                      Properties::ChildFlagBits::DEFAULT_OPEN)) {
                if (config.is_ui() && !data.parent && !descriptor.scratch && !descriptor.disabled &&
                    window_display_input->can_receive_from(output) &&
                    config.config_bool("Show in Window")) {
                    const std::string window =
                        add_node(registry.node_type_info<WindowNode>().node_type_name);
                    add_connection(data.identifier, window, descriptor.name,
                                   std::string{WindowNode::DISPLAY_INPUT});
                }

                std::vector<std::string> receivers;
                receivers.reserve(per_output_info.inputs.size());
                for (auto& [node, input] : per_output_info.inputs) {
                    const std::string& input_name =
                        node_data.at(node).input_descriptors.at(input).name;
                    receivers.emplace_back(
                        fmt::format("({}, {})", input_name, node_data.at(node).label));
                }

                const std::span<const ResourceData::Copy> copies =
                    per_output_info.resource_data
                        ? std::span<const ResourceData::Copy>(per_output_info.resource_data->copies)
                        : std::span<const ResourceData::Copy>();
                std::string current_copy_index = "none";
                if (!copies.empty() && !per_output_info.precomputed_resources.empty()) {
                    const uint32_t set_idx = data.set_index(run_iteration);
                    current_copy_index = fmt::format(
                        "{:02}", std::get<1>(per_output_info.precomputed_resources[set_idx]));
                }

                config.output_text(fmt::format("# Copies: {:02}\nCopy index: {}\nSending to: [{}]",
                                               copies.size(), current_copy_index,
                                               fmt::join(receivers, ", ")));

                if (config.st_begin_child("connector", "Connector Properties")) {
                    output->properties(config);
                    config.st_end_child();
                }
                if (config.st_begin_child("resources", "Resource Properties")) {
                    for (uint32_t i = 0; i < copies.size(); i++) {
                        if (config.st_begin_child(fmt::format("resource_{}", i),
                                                  fmt::format("Copy {:02}", i))) {
                            copies[i].resource->properties(config);
                            config.st_end_child();
                        }
                    }
                    config.st_end_child();
                }

                config.st_end_child();
            }
        }
        const std::map<std::string, Port> public_outputs(data.public_outputs.begin(),
                                                         data.public_outputs.end());
        for (const auto& [port, source] : public_outputs) {
            if (config.st_begin_child(fmt::format("{} (public)", port),
                                      fmt::format("{} (public)", port),
                                      Properties::ChildFlagBits::DEFAULT_OPEN)) {
                const NodeData& source_data = node_data.at(source.node);
                const auto connector = source_data.output_connector_for_name.find(source.name);
                const bool has_connector = connector != source_data.output_connector_for_name.end();
                if (config.is_ui() && !data.parent && has_connector &&
                    window_display_input->can_receive_from(connector->second) &&
                    config.config_bool("Show in Window")) {
                    const std::string window =
                        add_node(registry.node_type_info<WindowNode>().node_type_name);
                    add_connection(data.identifier, window, port,
                                   std::string{WindowNode::DISPLAY_INPUT});
                }
                std::vector<std::string> receivers;
                if (const auto connections =
                        has_connector ? source_data.output_connections.find(connector->second)
                                      : source_data.output_connections.end();
                    connections != source_data.output_connections.end()) {
                    for (const auto& [reader, input] : connections->second.inputs) {
                        receivers.emplace_back(fmt::format(
                            "({}, {})", node_data.at(reader).input_descriptors.at(input).name,
                            node_data.at(reader).label));
                    }
                }
                config.output_text("Carried by: {}.{}\nSending to: [{}]", source_data.identifier,
                                   source.name, fmt::join(receivers, ", "));
                config.st_end_child();
            }
        }
        config.st_end_child();
    }
    if (!needs_reconnect && (!data.input_connectors.empty() || !data.public_inputs.empty()) &&
        config.st_begin_child("inputs", "Inputs")) {
        for (const auto& input : data.input_connectors) {
            const std::string& input_name = data.input_descriptors.at(input).name;
            if (config.st_begin_child(input_name, input_name)) {
                config.st_separate("Input Properties");
                input->properties(config);
                config.st_separate("Connection");
                if (data.input_connections.contains(input)) {
                    auto& per_input_info = data.input_connections.at(input);
                    if (per_input_info.output) {
                        const std::string& src_output_name =
                            node_data.at(per_input_info.node)
                                .output_descriptors.at(per_input_info.output)
                                .name;
                        config.output_text(fmt::format("Receiving from: {}, {}", src_output_name,
                                                       node_data.at(per_input_info.node).label));
                    } else {
                        config.output_text("Optional input not connected.");
                    }
                } else {
                    config.output_text("Input not connected.");
                }

                if (data.desired_incoming_connections.contains(input_name) &&
                    config.config_bool("Remove Connection")) {
                    remove_connection(data.desired_incoming_connections.at(input_name).node, node,
                                      input_name);
                }

                config.st_end_child();
            }
        }
        const std::map<std::string, std::vector<Port>> public_inputs(data.public_inputs.begin(),
                                                                     data.public_inputs.end());
        for (const auto& [port, fed_inputs] : public_inputs) {
            if (config.st_begin_child(fmt::format("{} (public)", port),
                                      fmt::format("{} (public)", port))) {
                std::vector<std::string> fed;
                for (const auto& [child, input] : fed_inputs) {
                    const NodeData& child_data = node_data.at(child);
                    fed.emplace_back(fmt::format(
                        "{}.{}{}", child_data.identifier, input,
                        child_data.public_feeds.contains(input) ? "" : " (its auto connection)"));
                }
                config.output_text("Feeds: [{}]", fmt::join(fed, ", "));
                if (const std::optional<Connection> wired = wiring(node, port)) {
                    config.output_text("Receiving from: {}, {}", wired->src_output,
                                       node_data.at(wired->src).label);
                    if (data.desired_incoming_connections.contains(port) &&
                        config.config_bool("Remove Connection")) {
                        remove_connection(wired->src, node, port);
                    }
                } else {
                    config.output_text("Input not connected.");
                }
                config.st_end_child();
            }
        }
        config.st_end_child();
    }
}

} // namespace merian
