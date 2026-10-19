#pragma once

// Python.h must precede the standard headers
#include "json_caster.hpp"

#include "merian-graph/graph/graph.hpp"
#include "merian-graph/nodes/host_sink/host_sink.hpp"
#include "merian-graph/nodes/host_source/host_source.hpp"
#include "merian/utils/defer.hpp"
#include "merian/vk/window/host_window.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <string_view>
#include <thread>
#include <unordered_map>

namespace merian::python {

namespace nb = nanobind;

inline unsigned long main_thread_ident = 0;

// Calls wait(slice) without the GIL until it returns true, or false once timeout passed; between
// slices a Python signal handler that raises (Ctrl-C) ends the wait with its exception.
bool wait_interruptibly(const std::function<bool(std::chrono::nanoseconds slice)>& wait,
                        const std::optional<std::chrono::nanoseconds>& timeout = {});

// While run() or a call that releases the GIL runs, only stop() and the host methods of the sink
// and source nodes are allowed from other threads; capture callbacks, which run on the running
// thread between iterations, may also edit the graph. Calls that wait for the GPU (load, run, wait,
// close, connect) need an idle graph.
class PythonGraph {
  public:
    PythonGraph(const ContextHandle& context,
                const std::optional<GraphDescription>& description,
                const bool strict);

    PythonGraph(const PythonGraph&) = delete;
    PythonGraph& operator=(const PythonGraph&) = delete;

    ~PythonGraph();

    void load(const GraphDescription& description, const bool strict);

    GraphRunResult run(const GraphRunOptions& options);

    void stop();

    void wait();

    void close();

    // The graph's settings, recorded into ui, a merian.Properties.
    void properties(const nb::object& ui);

    // The node's section, recorded into ui, a merian.Properties.
    void node_properties(const std::string& identifier, const nb::object& ui);

    void add_capture_callback(const std::shared_ptr<HostSink>& sink, const nb::callable& callback);

    void remove_capture_callbacks(const std::shared_ptr<HostSink>& sink);

    // Cancelled once the graph removes it, which ends the waits of consumer threads.
    std::shared_ptr<HostSink> sink(const std::string& identifier) const;

    std::shared_ptr<HostSource> source(const std::string& identifier) const;

    // The HostWindow of the Window node, null if it has none.
    std::shared_ptr<HostWindow> window(const std::string& identifier) const;

    std::unordered_map<std::string, std::shared_ptr<HostWindow>> windows() const;

    const GraphHandle& idle_handle() const;

    const GraphHandle& editable_handle() const;

    template <typename F> auto without_gil(const char* activity, const F& f) const {
        busy = activity;
        defer {
            busy = nullptr;
        };
        const nb::gil_scoped_release release;
        return f();
    }

    int visit_callbacks(visitproc visit, void* arg) const;
    void clear_callbacks();

  private:
    struct SinkCallbacks {
        std::shared_ptr<HostSink> sink;
        std::vector<nb::callable> callbacks;
    };

    void ensure_idle() const;

    void ensure_editable() const;

    const GraphHandle& handle() const;

    void record(const nb::object& ui, const std::function<void(Properties& props)>& pass);

    template <typename T>
    std::shared_ptr<T> find(const std::string& identifier, const std::string_view kind) const;

    void wait_for_iterations(const char* activity);

    // Needs the GIL.
    void dispatch_captures();

    bool captures_ready() const;

    static void check_problems(const std::vector<std::string>& problems, const bool strict);

    GraphHandle graph;
    std::atomic_bool running = false;
    mutable std::atomic<const char*> busy = nullptr;
    std::atomic<std::thread::id> dispatch_thread;
    std::atomic_bool stop_requested = false;
    // keeps those of a removed sink until dispatch_captures() delivered its captures
    std::vector<SinkCallbacks> capture_callbacks;
};

} // namespace merian::python
