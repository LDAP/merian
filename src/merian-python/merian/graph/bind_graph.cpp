// Python.h must precede the standard headers
#include "bindings.hpp"

#include <nanobind/stl/chrono.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>

namespace merian::python {

using namespace nb::literals;

namespace {

struct GraphProfiler : GraphReference {};

} // namespace

void bind_graph(nb::module_& m) {
    nb::enum_<GraphRunStopReason>(m, "StopReason")
        .value("ITERATIONS", GraphRunStopReason::ITERATIONS)
        .value("EVENT", GraphRunStopReason::EVENT)
        .value("DURATION", GraphRunStopReason::DURATION)
        .value("REQUESTED", GraphRunStopReason::REQUESTED, "stop() or an exception");

    nb::class_<GraphRunResult>(m, "RunResult")
        .def_ro("stop_reason", &GraphRunResult::stop_reason)
        .def_ro("iterations", &GraphRunResult::iterations)
        .def_ro("event", &GraphRunResult::event, "'node type/identifier/event' that ended the run")
        .def("__repr__", [](const GraphRunResult& result) {
            return fmt::format("RunResult(stop_reason={}, iterations={}, event='{}')",
                               nb::cast<std::string>(nb::str(nb::cast(result.stop_reason))),
                               result.iterations, result.event);
        });

    nb::class_<IterationInfo>(
        m, "IterationInfo",
        "An iteration as the callbacks of Graph.run see it: after_iteration the one that ran, "
        "before_iteration the one before the next.")
        .def_ro("index_in_run", &IterationInfo::index_in_run, "Of the iteration in this run.")
        .def_ro("iteration", &IterationInfo::iteration, "Since connect.")
        .def_ro("total_iteration", &IterationInfo::total_iteration)
        .def_ro("time", &IterationInfo::time, "Graph time in seconds.")
        .def_ro("time_delta", &IterationInfo::time_delta, "Seconds since the iteration before.")
        .def_ro("elapsed_since_connect", &IterationInfo::elapsed_since_connect, "Seconds.")
        .def("__repr__", [](const IterationInfo& info) {
            return fmt::format("IterationInfo(index_in_run={}, iteration={}, time={})",
                               info.index_in_run, info.iteration, info.time);
        });

    nb::enum_<ProfilerReportKind>(m, "ReportKind")
        .value("RUN", ProfilerReportKind::RUN, "Since the last connect or restart_report().")
        .value("INTERVAL", ProfilerReportKind::INTERVAL)
        .value("BUILD", ProfilerReportKind::BUILD, "The last connect.");

    nb::class_<GraphProfiler>(m, "GraphProfiler",
                              nb::type_slots(graph_reference_slots<GraphProfiler>))
        .def_prop_rw(
            "enabled",
            [](const GraphProfiler& profiler) {
                return profiler.python_graph().editable_handle()->profiler_get_enabled();
            },
            [](const GraphProfiler& profiler, const bool enabled) {
                profiler.python_graph().editable_handle()->profiler_set_enabled(enabled);
            },
            "Not stored. Off by default; while on, the nodes of a layer do not overlap on the "
            "GPU.")
        .def(
            "report",
            [](const GraphProfiler& profiler, const ProfilerReportKind kind) {
                return profiler.python_graph().editable_handle()->profiler_get_report(kind);
            },
            "kind"_a = ProfilerReportKind::RUN,
            "Of the iterations whose GPU timings were collected; Graph.wait() collects all.")
        .def(
            "restart_report",
            [](const GraphProfiler& profiler) {
                profiler.python_graph().editable_handle()->profiler_restart_report();
            },
            "Restarts the period of the RUN report.");

    nb::class_<GraphMemoryReport> memory_report(m, "MemoryReport");
    nb::enum_<GraphMemoryReport::View>(memory_report, "View")
        .value("SUMMARY", GraphMemoryReport::View::SUMMARY)
        .value("NODES", GraphMemoryReport::View::NODES)
        .value("LAYERS", GraphMemoryReport::View::LAYERS)
        .value("OUTPUTS", GraphMemoryReport::View::OUTPUTS)
        .value("BLOCKS", GraphMemoryReport::View::BLOCKS);
    memory_report
        .def_prop_ro("total_size", &GraphMemoryReport::total_size,
                     "Bytes of the outputs and of what the nodes allocate themselves.")
        .def(
            "to_text",
            [](const GraphMemoryReport& report, const std::optional<GraphMemoryReport::View> view) {
                return view ? report.to_text(*view) : report.to_text();
            },
            "view"_a = nb::none(), "One view, every view if None.")
        .def("to_json", &GraphMemoryReport::to_json)
        .def("__str__", [](const GraphMemoryReport& report) { return report.to_text(); });

    nb::class_<PythonGraph>(m, "Graph",
                            "While it runs, other threads may only call stop() and the methods "
                            "that HostSink, the sources and HostWindow document as callable from "
                            "any thread.")
        .def(nb::init<const ContextHandle&, const std::optional<GraphDescription>&, bool>(),
             "context"_a, "description"_a = nb::none(), "strict"_a = true,
             "strict raises ConfigError for properties of the description no node reads or with "
             "invalid values, else they log a warning.")
        .def("load", &PythonGraph::load, "description"_a, "strict"_a = true,
             "Replaces all nodes and connections after waiting for the iterations in flight; "
             "capture requests are removed.")
        .def(
            "run",
            [](PythonGraph& graph, const std::optional<uint64_t> iterations,
               const std::string& until_event, const std::optional<Seconds> duration,
               const std::optional<Seconds> time, const std::optional<nb::callable>& before,
               const std::optional<nb::callable>& after) {
                PythonGraph::BeforeIteration before_iteration;
                if (before) {
                    before_iteration = [&](const IterationInfo& info) {
                        const nb::object held = (*before)(info);
                        if (held.is_none()) {
                            return std::optional<std::chrono::nanoseconds>{};
                        }
                        Seconds seconds;
                        if (!nb::try_cast(held, seconds)) {
                            PyErr_SetString(PyExc_TypeError,
                                            "before_iteration must return None or a time");
                            throw nb::python_error();
                        }
                        return to_nanoseconds(seconds);
                    };
                }
                PythonGraph::AfterIteration after_iteration;
                if (after) {
                    after_iteration = [&](const IterationInfo& info) { (*after)(info); };
                }
                return graph.run(
                    {
                        .iterations = iterations,
                        .until_event = until_event,
                        .duration = to_limit(duration),
                        .time = to_nanoseconds(time),
                    },
                    before_iteration, after_iteration);
            },
            nb::arg("iterations").none() = 1, "until_event"_a = "", "duration"_a = nb::none(),
            "time"_a = nb::none(), "before_iteration"_a = nb::none(),
            "after_iteration"_a = nb::none(),
            "Runs until `iterations` ran (None: no limit), an event matching `until_event` "
            "(comma-separated 'node type/identifier/event' patterns; the graph sends "
            "'/graph/connect' and '/graph/reload_shaders') was sent, `duration` passed or stop() "
            "was called. `time` sets the graph time of every iteration of this run, over any time "
            "source, which continues from it in the next run without a time; durations and times "
            "are seconds or timedelta. With `iterations` the nodes see the last iteration, which "
            "sinks can write ('end of run'). Returns once the last iteration is submitted, after "
            "running the node callbacks (such as image writes) whose GPU work already finished; "
            "the rest run in the next run() or in wait(). before_iteration(info) is called before "
            "each iteration while the earlier ones may still be in flight, and may edit the graph "
            "and its scenes for it; returning a time (seconds or timedelta) holds the graph time "
            "from that iteration on, as `time` does. after_iteration(info) is called after each "
            "iteration is submitted; sink.get_all() there returns the captures that arrived. An "
            "exception of a callback "
            "ends the run before the next iteration and is raised from run().")
        .def("stop", &PythonGraph::stop,
             "Ends a run after the current iteration, no effect between runs; callable from any "
             "thread.")
        .def("wait", &PythonGraph::wait,
             "Waits for all iterations in flight and their node callbacks (written image files "
             "and captures are complete afterwards).")
        .def("close", &PythonGraph::close,
             "Waits for the iterations in flight and releases the graph; captures not taken are "
             "dropped.")
        .def(
            "__enter__", [](PythonGraph& graph) -> PythonGraph& { return graph; },
            nb::rv_policy::reference)
        .def("__exit__",
             [](PythonGraph& graph, const nb::args& exception) {
                 if (exception[0].is_none()) {
                     try {
                         graph.wait();
                     } catch (...) {
                         graph.close();
                         throw;
                     }
                 }
                 graph.close();
             })
        .def(
            "connect",
            [](const PythonGraph& graph) {
                const GraphHandle graph_handle = graph.idle_handle();
                graph.without_gil("connecting", [&] {
                    do {
                        graph_handle->connect();
                    } while (graph_handle->get_needs_reconnect());
                });
            },
            "Connects the graph, which run() also does; see inactive_nodes.")
        .def(
            "restart_run", [](const PythonGraph& graph) { graph.editable_handle()->restart_run(); },
            "Reconnects before the next iteration, which starts the run from scratch like any "
            "connect: run iteration 0, every history cleared, the RUN report started again.")
        .def(
            "reload_shaders",
            [](const PythonGraph& graph, const bool force) {
                const GraphHandle graph_handle = graph.editable_handle();
                graph.without_gil("reloading shaders",
                                  [&] { graph_handle->reload_shaders(force); });
            },
            "force"_a = false,
            "Recompiles the shaders whose files changed, includes and imports alike, all if "
            "force. Every connect, also the one run() does, reloads.")
        .def_prop_ro(
            "nodes",
            [](const nb::object& graph) {
                const GraphHandle& graph_handle = nb::cast<PythonGraph&>(graph).editable_handle();
                nb::dict nodes;
                for (const std::string& identifier : graph_handle->get_top_level_identifiers()) {
                    nodes[identifier.c_str()] = NodeByIdentifier{{graph}, identifier};
                }
                return nodes;
            },
            "{identifier: Node}, without the nodes of subgraphs.")
        .def_prop_ro(
            "inactive_nodes",
            [](const PythonGraph& graph) {
                nb::dict inactive;
                for (const auto& [identifier, reason] :
                     graph.editable_handle()->get_inactive_nodes()) {
                    inactive[identifier.c_str()] = nb::str(reason.c_str());
                }
                return inactive;
            },
            "{identifier: reason} of the enabled nodes that do not run since the last connect.")
        .def(
            "add_node",
            [](const PythonGraph& graph, const std::string& node_type,
               const std::optional<std::string>& identifier, const nlohmann::json& properties) {
                const GraphHandle graph_handle = graph.editable_handle();
                return graph.without_gil("adding a node", [&] {
                    const std::string added = graph_handle->add_node(node_type, identifier);
                    try {
                        if (!properties.empty()) {
                            graph_handle->node_set_properties(added, properties);
                        }
                    } catch (...) {
                        graph_handle->remove_node(added);
                        throw;
                    }
                    return added;
                });
            },
            "node_type"_a, "identifier"_a = nb::none(), "properties"_a = nlohmann::json::object(),
            "Returns the identifier; the graph connects it before the next iteration.")
        .def(
            "remove_node",
            [](const PythonGraph& graph, const std::string& identifier) {
                const GraphHandle graph_handle = graph.editable_handle();
                return graph.without_gil("removing a node",
                                         [&] { return graph_handle->remove_node(identifier); });
            },
            "identifier"_a, "A host sink is cancelled: its waiting get() calls return.")
        .def(
            "add_connection",
            [](const PythonGraph& graph, const std::string& src, const std::string& dst,
               const std::string& src_output, const std::string& dst_input) {
                graph.editable_handle()->add_connection(src, dst, src_output, dst_input);
            },
            "src"_a, "dst"_a, "src_output"_a, "dst_input"_a)
        .def(
            "remove_connection",
            [](const PythonGraph& graph, const std::string& src, const std::string& dst,
               const std::string& dst_input) {
                return graph.editable_handle()->remove_connection(src, dst, dst_input);
            },
            "src"_a, "dst"_a, "dst_input"_a)
        .def_prop_ro(
            "needs_reconnect",
            [](const PythonGraph& graph) { return graph.editable_handle()->get_needs_reconnect(); },
            "Whether an edit awaits connect(), which run() does first.")
        .def_prop_ro(
            "run_iteration",
            [](const PythonGraph& graph) { return graph.editable_handle()->get_run_iteration(); },
            "The iteration the next run processes, counted since the last connect: 0 while a "
            "reconnect is pending. Random numbers and histories follow it.")
        .def_prop_ro(
            "time",
            [](const PythonGraph& graph) {
                return std::chrono::duration<double>(graph.editable_handle()->get_time()).count();
            },
            "The graph time of the last iteration in seconds.")
        .def("properties", &PythonGraph::properties, "ui"_a,
             "Records the graph's settings into ui, a merian.Properties; what ui returns as set "
             "applies now, see needs_reconnect.")
        .def("get_properties",
             [](const PythonGraph& graph) { return graph.editable_handle()->get_properties(); })
        .def(
            "set_properties",
            [](const PythonGraph& graph, const nlohmann::json& properties) {
                const GraphHandle graph_handle = graph.editable_handle();
                graph.without_gil("setting properties",
                                  [&] { graph_handle->set_properties(properties); });
            },
            "properties"_a,
            "Sets the given settings, the others keep their value. Raises ValueError for unknown "
            "keys and invalid values; the valid ones stay set.")
        .def_prop_ro("profiler", [](const nb::object& graph) { return GraphProfiler{{graph}}; })
        .def("memory_report",
             [](const PythonGraph& graph) { return graph.editable_handle()->memory_get_report(); })
        .def(
            "send_event",
            [](const PythonGraph& graph, const std::string& name) {
                const GraphHandle graph_handle = graph.editable_handle();
                graph.without_gil("sending an event", [&] { graph_handle->send_event(name); });
            },
            "name"_a,
            "Sends the user event name. Nodes listen once the graph connected: connect() before "
            "the first run to reach them.")
        .def(
            "set_store_path",
            [](const PythonGraph& graph, const std::filesystem::path& path) {
                graph.editable_handle()->ui_set_store_path(path);
            },
            "path"_a, "The file Store writes and Load reads in the graph's ImGui panel.")
        .def("to_description",
             [](const PythonGraph& graph) { return graph.editable_handle()->to_description(); })
        .def(
            "find",
            [](const nb::object& graph, const nb::type_object& type,
               const std::string& identifier) {
                if (!nb::hasattr(type, "_find")) {
                    throw nb::type_error(
                        fmt::format("{} is not a node reference type", nb::type_name(type).c_str())
                            .c_str());
                }
                return type.attr("_find")(graph, identifier);
            },
            "type"_a, "identifier"_a,
            "The node as a reference of type, e.g. find(SceneNode, 'scene'); KeyError if the "
            "graph has no node of that type with this identifier.")
        .def(
            "sink",
            [](const nb::object& graph, const std::string& identifier) {
                return NodeReference<HostSink>{
                    {graph}, identifier, nb::cast<PythonGraph&>(graph).sink(identifier)};
            },
            "identifier"_a)
        .def(
            "source",
            [](const nb::object& graph, const std::string& identifier) -> nb::object {
                const std::shared_ptr<HostSource> source =
                    nb::cast<PythonGraph&>(graph).source(identifier);
                if (const auto image = std::dynamic_pointer_cast<HostImageSource>(source)) {
                    return nb::cast(NodeReference<HostImageSource>{{graph}, identifier, image});
                }
                return nb::cast(NodeReference<HostBufferSource>{
                    {graph}, identifier, std::dynamic_pointer_cast<HostBufferSource>(source)});
            },
            "identifier"_a, "The Host Image or Buffer Source.")
        .def("window", &PythonGraph::window, "identifier"_a,
             "The HostWindow of the Window node once connected, None if it has none: before the "
             "connect, in a context without HOST_WINDOWS_EXTENSION, or when its backend is "
             "another window provider.")
        .def_prop_ro(
            "windows", &PythonGraph::windows,
            "{identifier: HostWindow} of the Window nodes that have one (see window()), those in "
            "subgraphs included.");
}

} // namespace merian::python
