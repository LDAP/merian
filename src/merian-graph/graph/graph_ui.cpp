#include "merian-graph/graph/graph.hpp"

#include <algorithm>
#include <cstddef>
#include <fmt/chrono.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <vector>

namespace merian {

namespace {

// Auto-scaled time plots reach this quantile of their samples, times the headroom
constexpr float SCALE_QUANTILE = 0.95f;
constexpr float SCALE_HEADROOM = 1.1f;

// level keeps the ids of the CPU and the GPU tree apart
void draw_profiler_tree(Properties& props,
                        const std::vector<ProfilerReport::Entry>& entries,
                        const uint32_t level = 0) {
    for (const ProfilerReport::Entry& entry : entries) {
        const std::string text = entry.to_text();
        if (entry.children.empty()) {
            // aligned with the child item symbol
            props.output_text(fmt::format("   {}", text));
        } else if (props.st_begin_child(fmt::format("{}-{}", level, entry.name), text)) {
            draw_profiler_tree(props, entry.children, level + 1);
            props.st_end_child();
        }
    }
}

} // namespace

void Graph::draw_ui(Properties& props) {
    {
        const BusyScope busy_scope(*this);
        draw_controls(props);
        if (props.st_begin_child("graph_properties", "Graph Properties",
                                 Properties::ChildFlagBits::FRAMED)) {
            draw_state(props);
            graph_settings(props);
            props.st_end_child();
        }
        if (props.st_begin_child("edit", "Edit Graph", Properties::ChildFlagBits::FRAMED)) {
            draw_edit(props);
            props.st_end_child();
        }
        if (props.st_begin_child("profiler", "Profiler", Properties::ChildFlagBits::FRAMED)) {
            draw_profiler(props);
            props.st_end_child();
        }
        if (props.st_begin_child("memory", "Memory", Properties::ChildFlagBits::FRAMED)) {
            draw_memory(props);
            props.st_end_child();
        }
        if (props.st_begin_child("nodes", "Nodes",
                                 Properties::ChildFlagBits::DEFAULT_OPEN |
                                     Properties::ChildFlagBits::FRAMED)) {
            for (const NodeHandle& node : top_level_nodes()) {
                node_section(props, node, true);
            }
            props.st_end_child();
        }
    }
    apply_deferred_edits();
}

void Graph::ui_set_store_path(const std::filesystem::path& path) {
    ui.store_path = path.string();
}

void Graph::draw_controls(Properties& props) {
    if (props.config_bool("Rebuild")) {
        request_reconnect();
    }
    if (props.config_bool("reload shaders",
                          "Recompile shaders whose source files changed on disk.")) {
        reload_shaders();
    }
    props.st_no_space();
    if (props.config_bool("reload shaders (force)",
                          "Recompile all shaders unconditionally, ignoring mtimes and caches.")) {
        reload_shaders(true);
    }

    props.st_separate("Load / Store");
    static_cast<void>(props.config_text("path", ui.store_path, false, "Path for Load and Store."));
    if (props.config_bool("Store") && !ui.store_path.empty()) {
        to_description().to_file(ui.store_path);
        SPDLOG_INFO("stored graph with {} nodes to {}", node_for_identifier.size(), ui.store_path);
    }
    props.st_no_space();
    if (props.config_bool("Load") && !ui.store_path.empty()) {
        when_idle([this, path = ui.store_path] {
            try {
                from_description(GraphDescription::from_file(path));
            } catch (const std::exception& e) {
                SPDLOG_ERROR("cannot load {}: {}", path, e.what());
            }
        });
    }

    props.st_separate("Events");
    if (props.config_text("send event", ui.send_event, true) && !ui.send_event.empty()) {
        try {
            send_event(ui.send_event);
        } catch (const std::invalid_argument& e) {
            SPDLOG_ERROR("{}", e.what());
        }
        ui.send_event.clear();
    }
}

void Graph::draw_state(Properties& props) {
    props.output_text("Run iteration: {}", run_iteration);
    props.output_text("Run Elapsed: {:%H:%M:%S}s", run_info.elapsed_since_connect);
    props.output_text("Total Elapsed: {:%H:%M:%S}s", run_info.elapsed);
    props.output_text("Time delta: {:04f}ms", to_milliseconds(run_info.time_delta));
    props.output_text("GPU wait: {:04f}ms", to_milliseconds(gpu_wait_time));
    props.output_text("Iterations in flight {:02}/{:02}",
                      total_iteration - iteration_semaphore->get_counter_value(), in_flight.size());
    if (settings.limit_fps > 0) {
        props.output_text("CPU sleep time: {:04f}ms", to_milliseconds(cpu_sleep_time));
    }
    if (!layers.empty()) {
        props.output_text("dependency layers: {}", layers.size());
    }

    const auto now = std::chrono::steady_clock::now();
    std::erase_if(recent_events, [&](const auto& entry) {
        return now - entry.second.time > std::chrono::seconds(3);
    });
    std::vector<std::string> events;
    for (const auto& [name, event] : recent_events) {
        events.emplace_back(fmt::format("{} (it {})", name, event.iteration));
    }
    props.output_text("Last events: {}",
                      events.empty() ? "<none>" : fmt::format("{}", fmt::join(events, ", ")));
    props.st_separate();
}

void Graph::draw_edit(Properties& props) {
    props.st_separate("Add Node");
    props.config_options("new type", ui.new_node_selected, registry.node_names(),
                         Properties::OptionsStyle::COMBO);
    if (props.config_text("new identifier", ui.new_node_identifier, true,
                          "Set an optional name for the node and press enter.") ||
        props.config_bool("Add Node")) {
        std::optional<std::string> optional_identifier;
        if (!ui.new_node_identifier.empty()) {
            optional_identifier = ui.new_node_identifier;
        }
        try {
            add_node(registry.node_names()[ui.new_node_selected], optional_identifier);
        } catch (const std::invalid_argument& e) {
            SPDLOG_ERROR("cannot add node: {}", e.what());
        }
    }
    props.output_text("{}: {}", registry.node_names()[ui.new_node_selected],
                      registry.node_info(registry.node_names()[ui.new_node_selected]).description);

    // children are edited through their parent
    const std::vector<std::string> node_ids = get_top_level_identifiers();
    if (!node_ids.empty()) {
        const int last_node = static_cast<int>(node_ids.size()) - 1;
        ui.add_connection_selected_src = std::min(ui.add_connection_selected_src, last_node);
        ui.add_connection_selected_dst = std::min(ui.add_connection_selected_dst, last_node);
        ui.remove_node_selected = std::min(ui.remove_node_selected, last_node);
        props.st_separate("Add Connection");
        bool autodetect_dst_input = false;
        if (props.config_options("connection src", ui.add_connection_selected_src, node_ids,
                                 Properties::OptionsStyle::COMBO)) {
            ui.add_connection_selected_src_output = 0;
            autodetect_dst_input = true;
        }
        std::vector<std::string> src_outputs;
        const NodeData& src_data =
            node_data.at(node_for_identifier.at(node_ids[ui.add_connection_selected_src]));
        for (const auto& [output_name, output] : src_data.output_connector_for_name) {
            if (!src_data.output_descriptors.at(output).scratch) {
                src_outputs.emplace_back(output_name);
            }
        }
        for (const auto& [public_output, source] : src_data.public_outputs) {
            src_outputs.emplace_back(public_output);
        }
        std::ranges::sort(src_outputs);
        autodetect_dst_input |=
            props.config_options("connection src output", ui.add_connection_selected_src_output,
                                 src_outputs, Properties::OptionsStyle::COMBO);
        if (props.config_options("connection dst", ui.add_connection_selected_dst, node_ids,
                                 Properties::OptionsStyle::COMBO)) {
            ui.add_connection_selected_dst_input = 0;
            autodetect_dst_input = true;
        }
        const NodeData& dst_data =
            node_data.at(node_for_identifier.at(node_ids[ui.add_connection_selected_dst]));
        std::vector<std::string> dst_inputs;
        dst_inputs.reserve(dst_data.input_connector_for_name.size());
        for (const auto& [input_name, input] : dst_data.input_connector_for_name) {
            dst_inputs.emplace_back(input_name);
        }
        for (const auto& [public_input, inputs] : dst_data.public_inputs) {
            if (!dst_data.input_connector_for_name.contains(public_input)) {
                dst_inputs.emplace_back(public_input);
            }
        }
        std::ranges::sort(dst_inputs);
        if (autodetect_dst_input &&
            ui.add_connection_selected_src_output < static_cast<int>(src_outputs.size())) {
            // an input named like the output
            for (int i = 0; i < static_cast<int>(dst_inputs.size()); i++) {
                if (dst_inputs[i] == src_outputs[ui.add_connection_selected_src_output]) {
                    ui.add_connection_selected_dst_input = i;
                }
            }
        }
        props.config_options("connection dst input", ui.add_connection_selected_dst_input,
                             dst_inputs, Properties::OptionsStyle::COMBO);
        const bool valid_connection =
            ui.add_connection_selected_src_output < static_cast<int>(src_outputs.size()) &&
            ui.add_connection_selected_dst_input < static_cast<int>(dst_inputs.size());
        if (valid_connection) {
            if (props.config_bool("Add Connection")) {
                add_connection(node_ids[ui.add_connection_selected_src],
                               node_ids[ui.add_connection_selected_dst],
                               src_outputs[ui.add_connection_selected_src_output],
                               dst_inputs[ui.add_connection_selected_dst_input]);
            }

            const auto it = dst_data.desired_incoming_connections.find(
                dst_inputs[ui.add_connection_selected_dst_input]);
            if (it != dst_data.desired_incoming_connections.end()) {
                props.st_no_space();
                props.output_text("Warning: Input already connected with {}, {}", it->second.name,
                                  node_data.at(it->second.node).label);
            }
        }
        props.st_separate("Remove Node");
        props.config_options("remove identifier", ui.remove_node_selected, node_ids,
                             Properties::OptionsStyle::COMBO);
        if (props.config_bool("Remove Node")) {
            remove_node(node_ids[ui.remove_node_selected]);
        }
    }
}

void Graph::draw_profiler(Properties& props) {
#ifdef MERIAN_PROFILER_ENABLE
    bool enabled = profiler_enable;
    if (props.config_bool("profiling", enabled)) {
        profiler_set_enabled(enabled);
    }
#else
    props.output_text("Profiler disabled at compile-time!\n\n Enable with 'meson configure "
                      "<builddir> -Dmerian:performance_profiling=true'.");
#endif

    if (profiler_enable) {
        props.st_no_space();
        profiler_settings(props);

        if (!interval_report.empty() &&
            props.st_begin_child("run", "Graph Run", Properties::ChildFlagBits::DEFAULT_OPEN)) {
            const auto timeline = [&](const std::string& title, const std::string& id,
                                      const std::vector<ProfilerReport::Entry>& entries,
                                      const RingBuffer<float>& history, float& scale_max,
                                      bool& auto_scale, const uint32_t level) {
                if (entries.empty()) {
                    return;
                }
                props.st_separate(title);
                const float* samples = &history[ui.time_history_current + 1];
                if (auto_scale) {
                    std::vector<float> sorted(samples, samples + history.size() - 1);
                    const auto quantile =
                        sorted.begin() +
                        static_cast<std::ptrdiff_t>(SCALE_QUANTILE *
                                                    static_cast<float>(sorted.size() - 1));
                    std::nth_element(sorted.begin(), quantile, sorted.end());
                    scale_max = *quantile * SCALE_HEADROOM;
                }
                props.output_plot_line(id, samples, history.size() - 1, 0, scale_max);
                auto_scale &= !props.config_float(fmt::format("{} max ms", id), scale_max, "", 0.1f,
                                                  0.0f, 1000.0f);
                props.st_no_space();
                props.config_bool(fmt::format("{} auto", id), auto_scale);
                draw_profiler_tree(props, entries, level);
            };
            timeline("CPU", "cpu", interval_report.cpu, ui.cpu_time_history, ui.cpu_max,
                     ui.cpu_auto, 0);
            timeline("GPU", "gpu", interval_report.gpu, ui.gpu_time_history, ui.gpu_max,
                     ui.gpu_auto, 1u << 31);
            props.st_end_child();
        }
    }
    if (!build_report.empty() && props.st_begin_child("build", "Last Graph Build")) {
        if (!build_report.cpu.empty()) {
            props.st_separate("CPU");
            draw_profiler_tree(props, build_report.cpu);
        }
        if (!build_report.gpu.empty()) {
            props.st_separate("GPU");
            draw_profiler_tree(props, build_report.gpu, 1u << 31);
        }
        props.st_end_child();
    }
}

void Graph::draw_memory(Properties& props) {
    const auto now = std::chrono::steady_clock::now();
    if (ui.memory_view_dirty || now - ui.memory_view_time > std::chrono::seconds(1)) {
        ui.memory_view_dirty = false;
        ui.memory_view_time = now;
        const GraphMemoryReport report = memory_get_report();
        ui.memory_view = {
            .summary = report.to_text(GraphMemoryReport::View::SUMMARY),
            .nodes = report.to_text(GraphMemoryReport::View::NODES),
            .layers = report.to_text(GraphMemoryReport::View::LAYERS),
            .outputs = report.to_text(GraphMemoryReport::View::OUTPUTS),
            .blocks = report.to_text(GraphMemoryReport::View::BLOCKS),
        };
    }

    props.output_text(ui.memory_view.summary);
    if (!props.st_begin_child("views", "", Properties::ChildFlagBits::TABS)) {
        return;
    }
    const auto tab = [&](const std::string& id, const std::string& label, const std::string& text) {
        if (!text.empty() && props.st_begin_child(id, label)) {
            props.output_text(text);
            props.st_end_child();
        }
    };
    tab("nodes", "Nodes", ui.memory_view.nodes);
    tab("layers", "Layers", ui.memory_view.layers);
    tab("outputs", "Outputs", ui.memory_view.outputs);
    tab("blocks", "Blocks", ui.memory_view.blocks);
    props.st_end_child();
}

} // namespace merian
