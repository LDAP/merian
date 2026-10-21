#include "python_graph.hpp"

#include "merian-graph/merian_graph_extension.hpp"
#include "merian-graph/nodes/window/window_node.hpp"
#include "merian/vk/extension/extension_resources.hpp"
#include "python_properties.hpp"

namespace merian::python {

namespace {

constexpr std::chrono::nanoseconds SIGNAL_CHECK_INTERVAL = std::chrono::milliseconds(100);

} // namespace

bool wait_interruptibly(const std::function<bool(std::chrono::nanoseconds slice)>& wait,
                        const std::optional<std::chrono::nanoseconds>& timeout) {
    const auto start = std::chrono::steady_clock::now();
    while (true) {
        const std::chrono::nanoseconds remaining =
            timeout ? *timeout - (std::chrono::steady_clock::now() - start) : SIGNAL_CHECK_INTERVAL;
        bool done = false;
        {
            const nb::gil_scoped_release release;
            done = wait(
                std::clamp(remaining, std::chrono::nanoseconds::zero(), SIGNAL_CHECK_INTERVAL));
        }
        if (done) {
            return true;
        }
        if (remaining <= SIGNAL_CHECK_INTERVAL && timeout) {
            return false;
        }
        if (PyErr_CheckSignals() != 0) {
            throw nb::python_error();
        }
    }
}

PythonGraph::PythonGraph(const ContextHandle& context,
                         const std::optional<GraphDescription>& description,
                         const bool strict) {
    const auto extension = context->get_context_extension<MerianGraphExtension>(true);
    if (!extension) {
        throw MissingExtension(fmt::format("the context lacks the '{}' extension, create it with "
                                           "extensions=merian.graph.EXTENSIONS",
                                           MerianGraphExtension::name));
    }
    const ResourceAllocatorHandle allocator =
        context->get_context_extension<ExtensionResources>()->resource_allocator();

    std::vector<std::string> problems;
    {
        const nb::gil_scoped_release release;
        graph = extension->create({.context = context, .resource_allocator = allocator});
        if (description) {
            problems = graph->from_description(*description);
        }
    }
    check_problems(problems, strict);
}

PythonGraph::~PythonGraph() {
    // destroying the graph waits for the GPU and runs node callbacks
    const nb::gil_scoped_release release;
    graph.reset();
}

void PythonGraph::load(const GraphDescription& description, const bool strict) {
    wait();
    const GraphHandle graph_handle = handle();
    const std::vector<std::string> problems =
        without_gil("loading", [&] { return graph_handle->from_description(description); });
    check_problems(problems, strict);
}

GraphRunResult PythonGraph::run(const GraphRunOptions& options,
                                const BeforeIteration& before_iteration,
                                const AfterIteration& after_iteration) {
    const GraphHandle graph_handle = idle_handle();
    running = true;
    stop_requested = false;
    defer {
        running = false;
    };

    const bool check_signals = PyThread_get_thread_ident() == main_thread_ident;
    auto next_signal_check = std::chrono::steady_clock::now() + SIGNAL_CHECK_INTERVAL;
    std::optional<nb::python_error> error;
    const auto signal_raised = [&] {
        const auto now = std::chrono::steady_clock::now();
        if (!check_signals || now < next_signal_check) {
            return false;
        }
        next_signal_check = now + SIGNAL_CHECK_INTERVAL;
        const nb::gil_scoped_acquire acquire;
        if (PyErr_CheckSignals() != 0) {
            error.emplace();
        }
        return error.has_value();
    };
    const auto info_of = [](const NodeProcessInfo& process) {
        return IterationInfo{
            .run_index = process.get_run_index(),
            .iteration = process.get_iteration(),
            .total_iteration = process.get_total_iteration(),
            .time = process.get_elapsed(),
            .time_delta = process.get_time_delta(),
            .elapsed_since_connect = process.get_elapsed_since_connect(),
        };
    };
    const auto before = [&](const NodeProcessInfo& info) {
        const nb::gil_scoped_acquire acquire;
        const DispatchScope dispatch(*this);
        try {
            return before_iteration(info_of(info));
        } catch (nb::python_error& e) {
            error.emplace(std::move(e));
        }
        return std::optional<std::chrono::nanoseconds>{};
    };
    const auto after = [&](const NodeProcessInfo& info) {
        if (error) {
            return;
        }
        const nb::gil_scoped_acquire acquire;
        const DispatchScope dispatch(*this);
        try {
            after_iteration(info_of(info));
        } catch (nb::python_error& e) {
            error.emplace(std::move(e));
        }
    };
    GraphRunResult result;
    {
        const nb::gil_scoped_release release;
        result = graph_handle->run(
            options, before_iteration ? Graph::BeforeIteration(before) : Graph::BeforeIteration{},
            after_iteration ? Graph::AfterIteration(after) : Graph::AfterIteration{},
            [&] { return error.has_value() || stop_requested || signal_raised(); });
    }
    if (error) {
        throw std::move(*error);
    }
    return result;
}

void PythonGraph::stop() {
    stop_requested = true;
}

void PythonGraph::wait() {
    if (!graph) {
        return;
    }
    wait_for_iterations("waiting");
}

void PythonGraph::close() {
    if (!graph) {
        return;
    }
    ensure_idle();
    wait_for_iterations("closing");
    without_gil("closing", [&] { graph.reset(); });
}

void PythonGraph::wait_for_iterations(const char* activity) {
    const GraphHandle graph_handle = idle_handle();
    busy = activity;
    defer {
        busy = nullptr;
    };
    wait_interruptibly(
        [&](const std::chrono::nanoseconds slice) { return graph_handle->wait(slice); });
}

void PythonGraph::properties(const nb::object& ui) {
    const GraphHandle graph_handle = handle();
    record(ui, [&](Properties& props) { graph_handle->properties(props); });
}

void PythonGraph::node_properties(const std::string& identifier, const nb::object& ui) {
    const GraphHandle graph_handle = handle();
    record(ui, [&](Properties& props) { graph_handle->node_properties(identifier, props); });
}

void PythonGraph::record(const nb::object& ui, const std::function<void(Properties& props)>& pass) {
    ensure_editable();
    PythonProperties properties{ui};
    without_gil("recording properties", [&] { pass(properties); });
    properties.rethrow_error();
}

std::shared_ptr<HostSink> PythonGraph::sink(const std::string& identifier) const {
    return find<HostSink>(identifier, "host sink");
}

std::shared_ptr<HostSource> PythonGraph::source(const std::string& identifier) const {
    return find<HostSource>(identifier, "host source");
}

std::shared_ptr<HostWindow> PythonGraph::window(const std::string& identifier) const {
    return std::dynamic_pointer_cast<HostWindow>(
        find<WindowNode>(identifier, "window node")->get_window());
}

std::unordered_map<std::string, std::shared_ptr<HostWindow>> PythonGraph::windows() const {
    const GraphHandle& graph_handle = editable_handle();
    std::unordered_map<std::string, std::shared_ptr<HostWindow>> windows;
    for (const std::string& identifier : graph_handle->get_identifiers()) {
        const auto node = std::dynamic_pointer_cast<WindowNode>(
            graph_handle->find_node_for_identifier(identifier));
        if (auto window =
                node ? std::dynamic_pointer_cast<HostWindow>(node->get_window()) : nullptr) {
            windows.emplace(identifier, std::move(window));
        }
    }
    return windows;
}

void PythonGraph::ensure_idle() const {
    ensure_editable();
    if (running || dispatch_thread != std::thread::id{}) {
        throw std::runtime_error("the graph is running or calling iteration callbacks");
    }
}

void PythonGraph::ensure_editable() const {
    if (const char* activity = busy) {
        throw std::runtime_error(fmt::format("the graph is busy {}", activity));
    }
    if (running && dispatch_thread != std::this_thread::get_id()) {
        throw std::runtime_error("the graph is running, edit it from an iteration callback or "
                                 "stop() it first");
    }
}

const GraphHandle& PythonGraph::handle() const {
    if (!graph) {
        throw std::runtime_error("the graph is closed");
    }
    return graph;
}

const GraphHandle& PythonGraph::idle_handle() const {
    ensure_idle();
    return handle();
}

const GraphHandle& PythonGraph::editable_handle() const {
    ensure_editable();
    return handle();
}

void PythonGraph::check_problems(const std::vector<std::string>& problems, const bool strict) {
    if (strict && !problems.empty()) {
        throw graph_errors::config_error{problems};
    }
}

} // namespace merian::python
