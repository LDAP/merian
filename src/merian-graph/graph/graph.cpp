#include "merian-graph/graph/graph.hpp"
#include "merian-graph/connectors/buffer/vk_buffer_out.hpp"
#include "merian-graph/connectors/image/vk_image_out.hpp"
#include "merian-graph/merian_graph_extension.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/utils/defer.hpp"
#include "merian/utils/properties.hpp"
#include "merian/vk/extension/extension_registry.hpp"
#include "merian/vk/utils/check_result.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"

#include <algorithm>
#include <any>
#include <array>
#include <ranges>
#include <spdlog/spdlog.h>

namespace merian {

using namespace graph_internal;
using namespace std::literals::chrono_literals;

namespace {

const ContextHandle& check_requirements_and_get_context(const ContextHandle& context) {
    if (context->get_context_extension<MerianGraphExtension>(true) == nullptr) {
        throw graph_errors::graph_error{
            "The merian-graph context extension must be enabled and supported."};
    }
    return context;
}

} // namespace

Graph::Graph(const GraphCreateInfo& create_info)
    : context(check_requirements_and_get_context(create_info.context)),
      resource_allocator(create_info.resource_allocator),
      queue(context->get_queue(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute |
                               vk::QueueFlagBits::eTransfer)),
      async_runner([this](std::function<void()>&& task) { run_async(std::move(task)); }),
      registry(NodeRegistry::get_instance()),
      run_profiler(std::make_shared<merian::Profiler>(context)), run_info(resource_allocator),
      iteration_semaphore(TimelineSemaphore::create(context)) {

    resize_in_flight(settings.iterations_in_flight);
    debug_utils = context->get_context_extension<ExtensionVkDebugUtils>(true);
    time_connect_reference = time_reference = std::chrono::high_resolution_clock::now();
    context_extension = context->get_context_extension<MerianGraphExtension>();

    run_info.metadata = [this]() { return get_metadata(); };
    run_info.iteration_semaphore = iteration_semaphore;
    run_info.wait_for_iteration = [this](const uint64_t value) { wait_for_iteration(value, {}); };
}

Graph::~Graph() {
    wait();
    for (const auto& [node, data] : node_data) {
        node->on_removed();
    }
}

bool Graph::iterate(const uint64_t run_index,
                    const bool is_last,
                    const BeforeIteration& before_iteration,
                    const std::function<bool()>& stop_requested) {
    bool ran = false;
    {
        const BusyScope busy_scope(*this);
        // the slot is the only point to cancel the iteration
        ran = acquire_slot(stop_requested);
        if (ran) {
            process_iteration(run_index, is_last, before_iteration);
        }
        run_finished_callbacks();
    }
    apply_deferred_edits();
    return ran;
}

void Graph::process_iteration(const uint64_t run_index,
                              const bool is_last,
                              const BeforeIteration& before_iteration) {
    InFlightData& in_flight_data = in_flight[in_flight_index];
    in_flight_data.submission->reset(async_runner);

    const ProfilerHandle& profiler = prepare_profiler_for_run(in_flight_data);
    const ScopedDefaultProfiler scoped_default_profiler{profiler};
    const auto run_start = std::chrono::high_resolution_clock::now();

    // 1. connect, clock, host callback, pre-process
    const std::chrono::nanoseconds previous_elapsed = run_info.elapsed;
    bool first_pass = true;
    do {
        while (needs_reconnect) {
            connect_graph();
        }
        // the time source may be a node
        if (first_pass) {
            advance_time(previous_elapsed);
        }
        update_run_info(profiler, run_index, is_last);
        if (std::exchange(first_pass, false) && before_iteration) {
            if (const auto time = before_iteration(run_info)) {
                held_time = time;
                advance_time(previous_elapsed);
            }
            // its edits connect before the nodes pre-process
            if (needs_reconnect) {
                continue;
            }
        }
        pre_process_nodes(in_flight_data, profiler);
    } while (needs_reconnect);

    // 2. record and submit
    Submission& submission = *in_flight_data.submission;
    try {
        record_iteration(submission, profiler);
    } catch (...) {
        try {
            // the images already track the layouts of the collected barriers
            flush_barriers(submission);
            finish_iteration(in_flight_data);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("finishing the failed iteration failed: {}", e.what());
        }
        throw;
    }

    {
        MERIAN_PROFILE_SCOPE(profiler, "end run");
        finish_iteration(in_flight_data);
    }

    cpu_time = std::chrono::high_resolution_clock::now() - run_start;
}

bool Graph::acquire_slot(const std::function<bool()>& stop_requested) {
    if (settings.iterations_in_flight != in_flight.size()) {
        wait();
        resize_in_flight(settings.iterations_in_flight);
        request_reconnect();
    }

    Stopwatch sw_gpu_wait;
    const uint32_t next_index = (in_flight_index + 1) % in_flight.size();
    if (!wait_for_iteration(in_flight[next_index].iteration_value, stop_requested)) {
        return false;
    }
    in_flight_index = next_index;
    gpu_wait_time = gpu_wait_time * 0.9 + sw_gpu_wait.duration() * 0.1;

    // fps limiter
    cpu_sleep_time = std::chrono::duration<double>::zero();
    if (settings.limit_fps != 0) {
        cpu_sleep_time = std::max(std::chrono::duration<double>::zero(),
                                  1s / static_cast<double>(settings.limit_fps) -
                                      std::chrono::duration<double>(cpu_time));
    }
    if (cpu_sleep_time > 0s) {
        std::this_thread::sleep_for(cpu_sleep_time);
    }
    return true;
}

void Graph::update_run_info(const ProfilerHandle& profiler,
                            const uint64_t run_index,
                            const bool is_last) {
    shader_object_allocator->set_iteration(in_flight_index);

    run_info.shader_object_allocator = shader_object_allocator;
    run_info.profiler = profiler;
    run_info.iteration = run_iteration;
    run_info.total_iteration = total_iteration;
    run_info.run_index = run_index;
    run_info.last_iteration = is_last;
    run_info.in_flight_index = in_flight_index;
    run_info.iterations_in_flight = static_cast<uint32_t>(in_flight.size());
}

void Graph::pre_process_nodes(InFlightData& in_flight_data, const ProfilerHandle& profiler) {
    MERIAN_PROFILE_SCOPE(profiler, "Preprocess nodes");
    for (auto& layer : layers) {
        for (auto& node : layer.nodes) {
            NodeData& data = node_data.at(node);
            MERIAN_PROFILE_SCOPE(profiler, data.label);
            const uint32_t set_idx = data.set_index(run_iteration);
            Node::NodeStatusFlags flags = 0;
            collect_node_errors(data.errors_queued, [&] {
                flags = node->pre_process(data.resource_maps[set_idx], run_info);
            });
            if (!data.errors_queued.empty()) {
                SPDLOG_ERROR("preprocessing {} failed:\n - {}", data.label,
                             fmt::join(data.errors_queued, "\n   - "));
                request_reconnect();
            }
            if ((flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                SPDLOG_DEBUG("node {} requested reconnect in pre_process", data.identifier);
                request_reconnect();
            }
            if ((flags & Node::NodeStatusFlagBits::RESET_IN_FLIGHT_DATA) != 0u) {
                in_flight_data.in_flight_data[node].reset();
            }
            if ((flags & Node::NodeStatusFlagBits::REMOVE_NODE) != 0u) {
                remove_node(data.identifier);
            }
        }
    }
}

void Graph::record_iteration(Submission& submission, const ProfilerHandle& profiler) {
    SPDLOG_TRACE("starting run: iteration: {}", run_info.get_iteration());
    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, submission.get_cmd(), "Run nodes");
        const CommandBufferHandle& cmd = submission.get_cmd();
        for (const Layer& layer : layers) {
            const Layer::Set& set = layer.sets[run_iteration % period];
            for (const ImageHandle& image : set.discards) {
                image_barriers.push_back(image->barrier2(
                    vk::ImageLayout::eGeneral, FULL_MEMORY_BARRIER.srcAccessMask,
                    FULL_MEMORY_BARRIER.dstAccessMask, FULL_MEMORY_BARRIER.srcStageMask,
                    FULL_MEMORY_BARRIER.dstStageMask, VK_QUEUE_FAMILY_IGNORED,
                    VK_QUEUE_FAMILY_IGNORED,
                    all_levels_and_layers(format_aspect_flags(image->get_format())), true));
            }
            for (const NodeHandle& node : layer.nodes) {
                pre_process_connectors(submission, node, node_data.at(node));
            }
            flush_barriers(submission, set.barrier);

            // nodes of a layer overlap, unless profiled
            for (const NodeHandle& node : layer.nodes) {
                if (profiler) {
                    cmd->barrier(vk::MemoryBarrier2{vk::PipelineStageFlagBits2::eAllCommands,
                                                    {},
                                                    vk::PipelineStageFlagBits2::eAllCommands,
                                                    {}});
                }
                NodeData& data = node_data.at(node);
                if (debug_utils) {
                    debug_utils->cmd_begin_label(*cmd, data.label);
                }
                process_node(submission, node, data, profiler);
                if (debug_utils) {
                    debug_utils->cmd_end_label(*cmd);
                }
            }

            for (const NodeHandle& node : layer.nodes) {
                post_process_connectors(submission, node, node_data.at(node));
            }
            flush_barriers(submission);
        }
    }
}

bool Graph::wait_for_iteration(const uint64_t value, const std::function<bool()>& stop_requested) {
    constexpr auto poll_interval = 100ms;
    constexpr auto warn_after = 5s;
    const Stopwatch waiting;
    bool warned = false;
    while (true) {
        const auto next_callback = run_finished_callbacks();
        if (iteration_semaphore->get_counter_value() >= value) {
            return true;
        }

        std::array<vk::Semaphore, 2> semaphores{**iteration_semaphore};
        std::array<uint64_t, 2> values{value};
        uint32_t count = 1;
        if (next_callback) {
            semaphores[1] = **next_callback->first;
            values[1] = next_callback->second;
            count = 2;
        }
        const vk::Result result = context->get_device()->get_device().waitSemaphores(
            vk::SemaphoreWaitInfo{vk::SemaphoreWaitFlagBits::eAny, count, semaphores.data(),
                                  values.data()},
            std::chrono::duration_cast<std::chrono::nanoseconds>(poll_interval).count());
        if (result != vk::Result::eTimeout) {
            check_result(result, "failed waiting for an iteration");
            continue;
        }
        if (stop_requested && stop_requested()) {
            return false;
        }
        if (!warned && waiting.duration() > warn_after) {
            SPDLOG_WARN("iteration {} has not finished after {:.0f} s", value - 1,
                        waiting.seconds());
            warned = true;
        }
    }
}

std::optional<std::pair<TimelineSemaphoreHandle, uint64_t>> Graph::run_finished_callbacks() {
    for (uint32_t age = 0; age < in_flight.size(); age++) {
        const SubmissionHandle& submission = in_flight_submission(age);
        submission->run_finished_callbacks(async_runner);
        if (const auto next = submission->next_callback_wait()) {
            return next;
        }
    }
    return std::nullopt;
}

const SubmissionHandle& Graph::in_flight_submission(const uint32_t age) {
    return in_flight[(in_flight_index + 1 + age) % in_flight.size()].submission;
}

void Graph::resize_in_flight(const uint32_t size) {
    while (in_flight.size() < size) {
        in_flight.push_back({
            .submission = std::make_shared<Submission>(context, queue),
            .profiler_query_pool =
                std::make_shared<QueryPool<vk::QueryType::eTimestamp>>(context, 1024, true),
        });
    }
    in_flight.resize(size);
    in_flight_index = 0;
}

void Graph::run_host_round_trips() {
    const auto waits_for_host = [&] {
        return std::ranges::any_of(
            std::views::iota(0u, static_cast<uint32_t>(in_flight.size())),
            [&](const uint32_t age) { return in_flight_submission(age)->waits_for_host(); });
    };
    while (true) {
        const auto next_callback = run_finished_callbacks();
        if (!next_callback || !waits_for_host()) {
            return;
        }
        next_callback->first->wait(next_callback->second);
    }
}

void Graph::run_async(std::function<void()>&& task) {
    if (!async_tasks) {
        async_tasks = std::make_shared<ThreadPool>(settings.max_async_tasks, 1);
    }
    {
        const std::scoped_lock lock(async_mutex);
        async_running++;
    }
    async_tasks->submit<void>([this, task = std::move(task)] {
        try {
            task();
        } catch (const std::exception& e) {
            SPDLOG_ERROR("host callback failed: {}", e.what());
        } catch (...) {
            SPDLOG_ERROR("host callback failed");
        }
        {
            const std::scoped_lock lock(async_mutex);
            async_running--;
        }
        async_finished.notify_all();
    });
}

void Graph::finish_iteration(InFlightData& in_flight_data) {
    in_flight_data.iteration_value = ++total_iteration;
    ++run_iteration;
    in_flight_data.submission->add_signal_semaphore(iteration_semaphore,
                                                    in_flight_data.iteration_value);
    in_flight_data.submission->finish();
}

GraphRunResult Graph::run(const GraphRunOptions& options,
                          const BeforeIteration& before_iteration,
                          const AfterIteration& after_iteration,
                          const std::function<bool()>& stop_requested) {
    run_stop_patterns = GraphEvent::Pattern::parse(options.until_event);
    for (const GraphEvent::Pattern& pattern : run_stop_patterns) {
        warn_if_no_sender(pattern);
    }
    run_stop_event.reset();
    if (options.time) {
        held_time = options.time;
    } else {
        release_time();
    }
    defer {
        run_stop_patterns.clear();
    };

    const auto start = std::chrono::steady_clock::now();
    GraphRunResult result{
        .stop_reason = GraphRunStopReason::ITERATIONS, .iterations = 0, .event = {}};
    while (!options.iterations || result.iterations < *options.iterations) {
        const bool is_last = options.iterations && result.iterations + 1 == *options.iterations;
        if (!iterate(result.iterations, is_last, before_iteration, stop_requested)) {
            result.stop_reason = GraphRunStopReason::REQUESTED;
            break;
        }
        result.iterations++;

        if (after_iteration) {
            after_iteration(run_info);
        }
        if (stop_requested && stop_requested()) {
            result.stop_reason = GraphRunStopReason::REQUESTED;
            break;
        }
        if (run_stop_event) {
            result.stop_reason = GraphRunStopReason::EVENT;
            result.event = std::move(*run_stop_event);
            break;
        }
        if (options.duration && std::chrono::steady_clock::now() - start >= *options.duration) {
            result.stop_reason = GraphRunStopReason::DURATION;
            break;
        }
    }
    // otherwise they stall until the next run
    run_host_round_trips();
    return result;
}

bool Graph::wait(const std::optional<std::chrono::nanoseconds>& timeout) {
    SPDLOG_DEBUG("wait until all in-flight iterations have finished");
    const auto start = std::chrono::steady_clock::now();
    if (!wait_for_iteration(total_iteration, [&] {
            return timeout && std::chrono::steady_clock::now() - start >= *timeout;
        })) {
        return false;
    }
    for (uint32_t age = 0; age < in_flight.size(); age++) {
        in_flight_submission(age)->reset(async_runner);
    }
    {
        std::unique_lock lock(async_mutex);
        const auto idle = [&] { return async_running == 0; };
        if (!timeout) {
            async_finished.wait(lock, idle);
        } else if (!async_finished.wait_until(lock, start + *timeout, idle)) {
            return false;
        }
    }

    if (profiler_enable && busy == 0) {
        for (const InFlightData& slot : in_flight) {
            run_profiler->set_query_pool(slot.profiler_query_pool);
            run_profiler->collect(false, true);
        }
    }
    return true;
}

void Graph::clear() {
    wait();

    reset_connections();
    const std::vector<NodeHandle> removed(std::views::keys(node_data).begin(),
                                          std::views::keys(node_data).end());
    node_data.clear();
    node_for_identifier.clear();
    loaded_description = {};
    for (InFlightData& slot : in_flight) {
        slot.in_flight_data.clear();
    }
    for (const NodeHandle& node : removed) {
        node->on_removed();
    }

    request_reconnect();
}

NodeState Graph::node_get_state(const std::string& identifier) const {
    return node_data.at(node_for(identifier)).state();
}

const std::string& Graph::node_get_type(const std::string& identifier) const {
    return node_data.at(node_for(identifier)).type_name;
}

NodeConnectors Graph::node_get_connectors(const std::string& identifier) const {
    const NodeHandle node = node_for(identifier);
    const NodeData& data = node_data.at(node);
    NodeConnectors connectors;
    for (const InputConnectorHandle& input : data.input_connectors) {
        const InputConnectorDescriptor& descriptor = data.input_descriptors.at(input);
        connectors.inputs.push_back(
            {.name = descriptor.name, .delay = descriptor.delay, .optional = descriptor.optional});
    }
    std::vector<std::string> public_inputs;
    for (const std::string& name : data.public_inputs | std::views::keys) {
        if (!data.input_connector_for_name.contains(name)) {
            public_inputs.push_back(name);
        }
    }
    std::ranges::sort(public_inputs);
    for (const std::string& name : public_inputs) {
        std::optional<uint32_t> delay;
        bool optional = true;
        for (const auto& [child, child_input] : data.public_inputs.at(name)) {
            const NodeData& child_data = node_data.at(child);
            const auto connector = child_data.input_connector_for_name.find(child_input);
            if (connector == child_data.input_connector_for_name.end()) {
                delay = 0;
                optional = false;
                continue;
            }
            const InputConnectorDescriptor& descriptor =
                child_data.input_descriptors.at(connector->second);
            delay = std::min(delay.value_or(descriptor.delay), descriptor.delay);
            optional = optional && descriptor.optional;
        }
        connectors.inputs.push_back(
            {.name = name, .delay = delay.value_or(0), .optional = optional});
    }

    const auto kind = [](const OutputConnectorHandle& output) {
        if (std::dynamic_pointer_cast<VkImageOut>(output)) {
            return NodeConnectors::Output::Kind::IMAGE;
        }
        if (std::dynamic_pointer_cast<VkBufferOut>(output)) {
            return NodeConnectors::Output::Kind::BUFFER;
        }
        return NodeConnectors::Output::Kind::OTHER;
    };
    for (const OutputConnectorHandle& output : data.output_connectors) {
        const OutputConnectorDescriptor& descriptor = data.output_descriptors.at(output);
        if (!descriptor.scratch) {
            connectors.outputs.push_back({.name = descriptor.name, .kind = kind(output)});
        }
    }
    std::vector<std::string> public_outputs;
    for (const std::string& name : data.public_outputs | std::views::keys) {
        if (!data.output_connector_for_name.contains(name)) {
            public_outputs.push_back(name);
        }
    }
    std::ranges::sort(public_outputs);
    for (const std::string& name : public_outputs) {
        const ConnectorRef carried = carrier({node, name});
        const NodeData& child_data = node_data.at(carried.node);
        const auto output = child_data.output_connector_for_name.find(carried.name);
        connectors.outputs.push_back({.name = name,
                                      .kind = output == child_data.output_connector_for_name.end()
                                                  ? NodeConnectors::Output::Kind::OTHER
                                                  : kind(output->second)});
    }

    for (const auto& [input, source] : data.desired_incoming_connections) {
        connectors.connections.try_emplace(input, node_data.at(source.node).identifier,
                                           source.name);
    }
    return connectors;
}

void Graph::request_reconnect() {
    needs_reconnect = true;
}

void Graph::restart_run() {
    request_reconnect();
}

void Graph::reset_run() {
    run_iteration = 0;
    time_connect_reference = std::chrono::high_resolution_clock::now();
    run_info.elapsed_since_connect = 0ns;
    run_profiler->clear(Profiler::Period::TOTAL);
}

void Graph::reload_shaders(const bool force) {
    // an edited file of any loaded module, a node's own included, recompiles every node
    const bool sources_changed = context->get_shader_compile_context()->retire_stale_session();
    send_graph_event("reload_shaders", std::any(force || sources_changed));
}

void Graph::release_time() {
    if (held_time) {
        jump_time(*std::exchange(held_time, std::nullopt));
    }
}

void Graph::jump_time(const std::chrono::nanoseconds time) {
    run_info.elapsed_since_connect =
        std::max(run_info.elapsed_since_connect + (time - run_info.elapsed), 0ns);
    run_info.elapsed = time;
    // keep the references current so falling back to the system clock does not jump
    const auto now = std::chrono::high_resolution_clock::now();
    time_reference = now - run_info.elapsed;
    time_connect_reference = now - run_info.elapsed_since_connect;
}

std::chrono::nanoseconds Graph::get_time() const {
    return run_info.elapsed;
}

std::vector<NodeHandle> Graph::time_providers() const {
    std::vector<NodeHandle> providers;
    for (const auto& layer : layers) {
        for (const auto& node : layer.nodes) {
            // removed since the last connect
            if (!node_data.contains(node)) {
                continue;
            }
            const auto provider = std::dynamic_pointer_cast<TimeProvider>(node);
            if (provider && provider->provides_time()) {
                providers.emplace_back(node);
            }
        }
    }
    return providers;
}

void Graph::advance_time(const std::chrono::nanoseconds previous_elapsed) {
    std::optional<std::chrono::nanoseconds> elapsed;

    if (held_time) {
        elapsed = *held_time;
    } else if (settings.time_source == TIME_SOURCE_TIME ||
               settings.time_source == TIME_SOURCE_DELTA) {
        elapsed = previous_elapsed +
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::duration<double>(settings.time_overwrite_delta_ms / 1000.));
        if (settings.time_source == TIME_SOURCE_TIME) {
            settings.time_overwrite_delta_ms = 0;
        }
    } else if (settings.time_source != TIME_SOURCE_SYSTEM) {
        // an identifier that is no longer a live provider behaves like Auto, which takes the
        // first provider that answers
        const std::vector<NodeHandle> providers = time_providers();
        const auto selected =
            std::find_if(providers.begin(), providers.end(), [&](const NodeHandle& node) {
                return settings.time_source == node_data.at(node).identifier;
            });
        if (selected != providers.end()) {
            elapsed = std::dynamic_pointer_cast<TimeProvider>(*selected)->provide_time();
        } else {
            for (const NodeHandle& node : providers) {
                elapsed = std::dynamic_pointer_cast<TimeProvider>(node)->provide_time();
                if (elapsed) {
                    break;
                }
            }
        }
    }

    if (elapsed) {
        jump_time(std::max(*elapsed, 0ns));
    } else {
        const auto now = std::chrono::high_resolution_clock::now();
        run_info.elapsed = now - time_reference;
        run_info.elapsed_since_connect = now - time_connect_reference;
    }
    // the time since connect is 0 in the first iteration after it, whatever the source jumped
    if (run_iteration == 0) {
        run_info.elapsed_since_connect = 0ns;
        time_connect_reference = std::chrono::high_resolution_clock::now();
    }

    // a seek or a loop moves the source's time backwards; nodes integrate over the delta
    run_info.time_delta = std::max(run_info.elapsed - previous_elapsed, 0ns);
}

uint64_t Graph::get_total_iteration() const {
    return total_iteration;
}

uint64_t Graph::get_run_iteration() const {
    return needs_reconnect ? 0 : run_iteration;
}

bool Graph::get_needs_reconnect() const {
    return needs_reconnect;
}

std::ranges::keys_view<std::ranges::ref_view<const std::map<std::string, NodeHandle>>>
Graph::get_identifiers() const {
    return std::as_const(node_for_identifier) | std::ranges::views::keys;
}

std::vector<std::string> Graph::get_top_level_identifiers() const {
    std::vector<std::string> identifiers;
    std::ranges::transform(top_level_nodes(), std::back_inserter(identifiers),
                           [&](const NodeHandle& node) { return node_data.at(node).identifier; });
    return identifiers;
}

ProfilerHandle Graph::prepare_profiler_for_run(InFlightData& in_flight_data) {
    if (!profiler_enable) {
        interval_report = {};
        return nullptr;
    }

    auto report = run_profiler->set_collect_get_every(in_flight_data.profiler_query_pool,
                                                      settings.profiler_report_interval_ms,
                                                      settings.profiler_evict_after_ms);

    if (report) {
        interval_report = std::move(*report);
        ui.cpu_time_history.set(ui.time_history_current, interval_report.cpu_total_ms());
        ui.gpu_time_history.set(ui.time_history_current, interval_report.gpu_total_ms());
        ui.time_history_current++;
    }

    return run_profiler;
}

void Graph::pre_process_connectors(Submission& submission, const NodeHandle& node, NodeData& data) {
    const uint32_t set_idx = data.set_index(run_iteration);
    for (const auto& [input, per_input_info] : data.input_connections) {
        if (!per_input_info.node) {
            continue;
        }
        const GraphResourceHandle& resource = per_input_info.precomputed_resources[set_idx];
        if ((input->on_pre_process(submission, resource, node, image_barriers, buffer_barriers) &
             Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
            SPDLOG_DEBUG("input {} of {} requested a reconnect",
                         data.input_descriptors.at(input).name, data.label);
            request_reconnect();
        }
    }
    for (const auto& [output, per_output_info] : data.output_connections) {
        const GraphResourceHandle& resource = per_output_info.precomputed_resources[set_idx];
        // a version was prepared at the producer of its root
        if (!resource || per_output_info.version_of) {
            continue;
        }
        if ((output->on_pre_process(submission, resource, node, image_barriers, buffer_barriers) &
             Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
            SPDLOG_DEBUG("output {} of {} requested a reconnect",
                         data.output_descriptors.at(output).name, data.label);
            request_reconnect();
        }
    }
}

void Graph::process_node(Submission& submission,
                         const NodeHandle& node,
                         NodeData& data,
                         [[maybe_unused]] const ProfilerHandle& profiler) {
    MERIAN_PROFILE_SCOPE_GPU(profiler, submission.get_cmd(), data.label);
    Node::NodeStatusFlags flags = 0;
    collect_node_errors(data.errors_queued, [&] {
        flags =
            node->process(data.resource_maps[data.set_index(run_iteration)], run_info, submission);
    });
    if (!data.errors_queued.empty()) {
        SPDLOG_ERROR("executing {} failed:\n - {}", data.label,
                     fmt::join(data.errors_queued, "\n   - "));
        request_reconnect();
    }
    if ((flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT) != 0u) {
        SPDLOG_DEBUG("{} requested a reconnect in process", data.label);
        request_reconnect();
    }
    if ((flags & Node::NodeStatusFlagBits::REMOVE_NODE) != 0u) {
        remove_node(data.identifier);
    }
}

void Graph::post_process_connectors(Submission& submission,
                                    const NodeHandle& node,
                                    NodeData& data) {
    const uint32_t set_idx = data.set_index(run_iteration);
    for (const auto& [input, per_input_info] : data.input_connections) {
        if (!per_input_info.node) {
            continue;
        }
        const GraphResourceHandle& resource = per_input_info.precomputed_resources[set_idx];
        if ((input->on_post_process(submission, resource, node, image_barriers, buffer_barriers) &
             Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
            SPDLOG_DEBUG("input {} of {} requested a reconnect",
                         data.input_descriptors.at(input).name, data.label);
            request_reconnect();
        }
    }
    for (const auto& [output, per_output_info] : data.output_connections) {
        const GraphResourceHandle& resource = per_output_info.precomputed_resources[set_idx];
        if (!resource || per_output_info.version_of) {
            continue;
        }
        Connector::ConnectorStatusFlags flags = 0;
        try {
            flags = output->on_post_process(submission, resource, node, image_barriers,
                                            buffer_barriers);
        } catch (const graph_errors::connector_error& e) {
            throw graph_errors::connector_error{fmt::format("output {} of {}: {}",
                                                            data.output_descriptors.at(output).name,
                                                            data.label, e.what())};
        }
        if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
            SPDLOG_DEBUG("output {} of {} requested a reconnect",
                         data.output_descriptors.at(output).name, data.label);
            request_reconnect();
        }
    }
}

void Graph::flush_barriers(Submission& submission, const vk::MemoryBarrier2& memory_barrier) {
    if (!memory_barrier.dstStageMask && image_barriers.empty() && buffer_barriers.empty()) {
        return;
    }
    submission.get_cmd()->barrier(vk::DependencyInfo{{},
                                                     memory_barrier.dstStageMask ? 1u : 0u,
                                                     &memory_barrier,
                                                     static_cast<uint32_t>(buffer_barriers.size()),
                                                     buffer_barriers.data(),
                                                     static_cast<uint32_t>(image_barriers.size()),
                                                     image_barriers.data()});
    image_barriers.clear();
    buffer_barriers.clear();
}

} // namespace merian

REGISTER_CONTEXT_EXTENSION(merian::MerianGraphExtension, "merian-graph");
