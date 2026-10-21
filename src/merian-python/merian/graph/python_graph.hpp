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

// wait(slice) without the GIL until true or timeout; Ctrl-C ends it with its exception
bool wait_interruptibly(const std::function<bool(std::chrono::nanoseconds slice)>& wait,
                        const std::optional<std::chrono::nanoseconds>& timeout = {});

struct IterationInfo {
    uint64_t run_index;
    uint64_t iteration;
    uint64_t total_iteration;
    // seconds
    double time;
    double time_delta;
    double elapsed_since_connect;
};

// While running, other threads may only call stop() and the host sink and source methods.
class PythonGraph {
  public:
    // Called with the GIL held; a returned time holds the graph time from that iteration on.
    using BeforeIteration =
        std::function<std::optional<std::chrono::nanoseconds>(const IterationInfo&)>;
    using AfterIteration = std::function<void(const IterationInfo&)>;

    PythonGraph(const ContextHandle& context,
                const std::optional<GraphDescription>& description,
                const bool strict);

    PythonGraph(const PythonGraph&) = delete;
    PythonGraph& operator=(const PythonGraph&) = delete;

    ~PythonGraph();

    void load(const GraphDescription& description, const bool strict);

    // An exception of a callback ends the run before the next iteration and is raised from it.
    GraphRunResult run(const GraphRunOptions& options,
                       const BeforeIteration& before_iteration = {},
                       const AfterIteration& after_iteration = {});

    void stop();

    void wait();

    void close();

    // The graph's settings, recorded into ui, a merian.Properties.
    void properties(const nb::object& ui);

    // The node's section, recorded into ui, a merian.Properties.
    void node_properties(const std::string& identifier, const nb::object& ui);

    // Cancelled once the graph removes it, which ends the waits of consumer threads.
    std::shared_ptr<HostSink> sink(const std::string& identifier) const;

    std::shared_ptr<HostSource> source(const std::string& identifier) const;

    // The HostWindow of the Window node, null if it has none.
    std::shared_ptr<HostWindow> window(const std::string& identifier) const;

    std::unordered_map<std::string, std::shared_ptr<HostWindow>> windows() const;

    // Raises KeyError if the graph has no node with this identifier that is a T.
    template <typename T>
    std::shared_ptr<T> find(const std::string& identifier, const std::string_view kind) const {
        const auto node =
            std::dynamic_pointer_cast<T>(editable_handle()->find_node_for_identifier(identifier));
        if (!node) {
            throw nb::key_error(fmt::format("no {} '{}'", kind, identifier).c_str());
        }
        return node;
    }

    const GraphHandle& idle_handle() const;

    const GraphHandle& editable_handle() const;

    // Raises unless the graph may be edited from this thread now.
    void ensure_editable() const;

    template <typename F> auto without_gil(const char* activity, const F& f) const {
        busy = activity;
        defer {
            busy = nullptr;
        };
        const nb::gil_scoped_release release;
        return f();
    }

  private:
    // Marks this thread as the one calling back, which may edit the running graph.
    class DispatchScope {
      public:
        explicit DispatchScope(const PythonGraph& graph)
            : graph(graph), previous(graph.dispatch_thread.exchange(std::this_thread::get_id())) {}
        ~DispatchScope() {
            graph.dispatch_thread = previous;
        }
        DispatchScope(const DispatchScope&) = delete;
        DispatchScope& operator=(const DispatchScope&) = delete;

      private:
        const PythonGraph& graph;
        const std::thread::id previous;
    };

    void ensure_idle() const;

    const GraphHandle& handle() const;

    void record(const nb::object& ui, const std::function<void(Properties& props)>& pass);

    void wait_for_iterations(const char* activity);

    static void check_problems(const std::vector<std::string>& problems, const bool strict);

    GraphHandle graph;
    std::atomic_bool running = false;
    mutable std::atomic<const char*> busy = nullptr;
    mutable std::atomic<std::thread::id> dispatch_thread;
    std::atomic_bool stop_requested = false;
};

} // namespace merian::python
