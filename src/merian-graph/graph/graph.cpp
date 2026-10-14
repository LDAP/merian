#include "merian-graph/graph/graph.hpp"
#include "merian-graph/merian_graph_extension.hpp"
#include "merian/shader/glsl_compiler_provider.hpp"
#include "merian/utils/properties.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/extension/extension_registry.hpp"
#include "merian/vk/utils/subresource_ranges.hpp"

#include <any>
#include <spdlog/spdlog.h>

namespace merian {

using namespace merian;
using namespace graph_internal;
using namespace std::literals::chrono_literals;

const ContextHandle& check_requirements_and_get_context(const ContextHandle& context) {
    if (context->get_context_extension<MerianGraphExtension>(true) == nullptr) {
        throw graph_errors::graph_error{
            "The merian-graph context extension must be enabled and supported."};
    }
    return context;
}

Graph::Graph(const GraphCreateInfo& create_info)
    : context(check_requirements_and_get_context(create_info.context)),
      resource_allocator(create_info.resource_allocator),
      queue(context->get_queue(vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute |
                               vk::QueueFlagBits::eTransfer)),
      thread_pool(std::make_shared<ThreadPool>()),
      cpu_queue(std::make_shared<CPUQueue>(context, thread_pool)),
      registry(NodeRegistry::get_instance()),
      ring_fences(context,
                  2,
                  [this](const uint32_t /*index*/) {
                      InFlightData in_flight_data;
                      in_flight_data.submission =
                          std::make_shared<Submission>(context, queue, cpu_queue);
                      in_flight_data.profiler_query_pool =
                          std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(
                              context, 1024, true);
                      return in_flight_data;
                  }),
      run_profiler(std::make_shared<merian::Profiler>(context)), run_info(resource_allocator) {

    debug_utils = context->get_context_extension<ExtensionVkDebugUtils>(true);
    time_connect_reference = time_reference = std::chrono::high_resolution_clock::now();
    duration_elapsed = 0ns;
    context_extension = context->get_context_extension<MerianGraphExtension>();

    // An ImGui node sends imgui_event each frame with a Properties to render into.
    register_event_listener(
        "//", [this](const GraphEvent::Info& info, const GraphEvent::Data& data) {
            bool matches = false;
            split(imgui_event, ",",
                  [&](const std::string& event) { matches = matches || event == info.event_name; });
            if (matches) {
                if (Properties* const* props = std::any_cast<Properties*>(&data)) {
                    properties(**props);
                }
            }
            return false;
        });

    run_info.metadata = [this]() { return metadata(); };
}

Graph::~Graph() {
    wait();
}

void Graph::run() {
    {
        const BusyScope busy_scope(*this);
        run_once();
    }
    apply_deferred_edits();
}

void Graph::run_once() {
    // PREPARE RUN: wait for fence, release resources, reset cmd pool
    if (flush_thread_pool_at_run_start) {
        thread_pool->wait_empty();
    }

    if (desired_iterations_in_flight != ring_fences.size()) {
        // TODO: Move to connect but currently this is not possible since below we get a
        // reference to the inflight data but then resize which might invalidate the reference
        // (because the internal buffer is resized...)
        ring_fences.resize(desired_iterations_in_flight);
        request_reconnect();
    }

    // wait for the in-flight processing to finish
    Stopwatch sw_gpu_wait;
    InFlightData& in_flight_data = ring_fences.next_cycle_wait_get();
    gpu_wait_time = gpu_wait_time * 0.9 + sw_gpu_wait.duration() * 0.1;

    in_flight_data.cpu_sleep_time = 0ms;

    // FPS LIMITER
    if (limit_fps != 0) {
        in_flight_data.cpu_sleep_time =
            std::max(in_flight_data.cpu_sleep_time,
                     1s / (double)limit_fps - std::chrono::duration<double>(cpu_time));
    }

    if (in_flight_data.cpu_sleep_time > 0ms) {
        const auto last_cpu_sleep_time = in_flight_data.cpu_sleep_time;
        in_flight_data.cpu_sleep_time =
            std::min(in_flight_data.cpu_sleep_time,
                     std::chrono::duration<double>(last_cpu_sleep_time * 1.05 + 1ms));
        std::this_thread::sleep_for(in_flight_data.cpu_sleep_time);
    }

    in_flight_data.submission->reset();

    const ProfilerHandle& profiler = prepare_profiler_for_run(in_flight_data);
    const ScopedDefaultProfiler scoped_default_profiler{profiler};
    const auto run_start = std::chrono::high_resolution_clock::now();

    // CONNECT and PREPROCESS
    // the time source may be a node, so the topology has to exist before the clock advances
    while (needs_reconnect) {
        connect_graph();
    }
    advance_time();

    do {
        // While connection nodes can signalize that they need to reconnect
        while (needs_reconnect) {
            connect_graph();
        }

        shader_object_allocator->set_iteration(ring_fences.current_cycle_index());

        // run_iteration resets on connect; timeline values must increase, so start a fresh
        // semaphore.
        if (run_iteration == 0) {
            iteration_semaphore = TimelineSemaphore::create(context);
        }
        run_info.iteration_semaphore = iteration_semaphore;
        run_info.shader_object_allocator = shader_object_allocator;
        run_info.profiler = profiler;
        run_info.iteration = run_iteration;
        run_info.total_iteration = total_iteration;
        run_info.in_flight_index = ring_fences.current_cycle_index();
        run_info.iterations_in_flight = ring_fences.size();
        run_info.time_delta = time_delta;
        run_info.elapsed = duration_elapsed;
        run_info.elapsed_since_connect = duration_elapsed_since_connect;

        // While preprocessing nodes can signalize that they need to reconnect as well
        {
            MERIAN_PROFILE_SCOPE(profiler, "Preprocess nodes");
            for (auto& layer : layers)
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
    } while (needs_reconnect);

    // RUN
    Submission& submission = *in_flight_data.submission;
    {
        MERIAN_PROFILE_SCOPE(profiler, "on_run_starting");
        SPDLOG_TRACE("starting run: iteration: {}", run_info.get_iteration());
        on_run_starting(run_info);
    }
    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, submission.get_cmd(), "Run nodes");
        const CommandBufferHandle& cmd = submission.get_cmd();
        // left over if a connector threw
        image_barriers.clear();
        buffer_barriers.clear();
        for (const Layer& layer : layers) {
            const Layer::Set& set = layer.sets[run_iteration % period];
            for (const ImageHandle& image : set.discards) {
                constexpr vk::AccessFlags2 all_access =
                    vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead;
                constexpr vk::PipelineStageFlags2 all_stages =
                    vk::PipelineStageFlagBits2::eAllCommands;
                image_barriers.push_back(image->barrier2(
                    vk::ImageLayout::eGeneral, all_access, all_access, all_stages, all_stages,
                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
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

    // FINISH RUN: submit

    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, submission.get_cmd(), "on_pre_submit");
        on_pre_submit(run_info);
    }

    {

        MERIAN_PROFILE_SCOPE(profiler, "end run");
        submission.add_signal_semaphore(iteration_semaphore, run_iteration + 1);
        submission.finish(ring_fences.reset());
    }
    {
        MERIAN_PROFILE_SCOPE(profiler, "on_post_submit");
        on_post_submit();
    }

    ++run_iteration;
    ++total_iteration;

    cpu_time = std::chrono::high_resolution_clock::now() - run_start;
}

void Graph::wait() {
    SPDLOG_DEBUG("wait until all in-flight iterations have finished");
    ring_fences.wait_all();
    cpu_queue->wait_idle();
}

void Graph::reset() {
    wait();

    reset_connections();
    node_data.clear();
    node_for_identifier.clear();
    loaded_description = {};
    for (uint32_t i = 0; i < ring_fences.size(); i++) {
        InFlightData& in_flight_data = ring_fences.get(i).user_data;
        in_flight_data.in_flight_data.clear();
    }

    needs_reconnect = true;
}

NodeState Graph::get_node_state(const std::string& identifier) const {
    return node_data.at(node_for(identifier)).state();
}

void Graph::request_reconnect() {
    needs_reconnect = true;
}

void Graph::set_time_source_delta(const float delta_ms) {
    time_source = TIME_SOURCE_DELTA;
    time_overwrite_delta_ms = delta_ms;
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

void Graph::advance_time() {
    const std::chrono::nanoseconds last_elapsed = duration_elapsed;
    std::optional<std::chrono::nanoseconds> elapsed;

    if (time_source == TIME_SOURCE_TIME || time_source == TIME_SOURCE_DELTA) {
        elapsed =
            duration_elapsed + std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::duration<double>(time_overwrite_delta_ms / 1000.));
        if (time_source == TIME_SOURCE_TIME) {
            time_overwrite_delta_ms = 0;
        }
    } else if (time_source != TIME_SOURCE_SYSTEM) {
        // an identifier that is no longer a live provider behaves like Auto, which takes the
        // first provider that answers
        const std::vector<NodeHandle> providers = time_providers();
        const auto selected =
            std::find_if(providers.begin(), providers.end(), [&](const NodeHandle& node) {
                return time_source == node_data.at(node).identifier;
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
        const std::chrono::nanoseconds next = std::max(*elapsed, 0ns);
        duration_elapsed_since_connect =
            std::max(duration_elapsed_since_connect + (next - duration_elapsed), 0ns);
        duration_elapsed = next;
        // keep the references current so falling back to the system clock does not jump
        const auto now = std::chrono::high_resolution_clock::now();
        time_reference = now - duration_elapsed;
        time_connect_reference = now - duration_elapsed_since_connect;
    } else {
        const auto now = std::chrono::high_resolution_clock::now();
        duration_elapsed = now - time_reference;
        duration_elapsed_since_connect = now - time_connect_reference;
    }

    // a seek or a loop moves the source's time backwards; nodes integrate over the delta
    time_delta = std::max(duration_elapsed - last_elapsed, 0ns);
}

void Graph::set_profiler_enabled([[maybe_unused]] const bool enabled) {
#ifdef MERIAN_PROFILER_ENABLE
    profiler_enable = enabled;
#endif
}

void Graph::set_profiler_report_interval(const uint32_t millis) {
    profiler_report_intervall_ms = millis;
}

Profiler::Report Graph::get_run_report() {
    if (!profiler_enable) {
        return {};
    }
    return run_profiler->get_report();
}

const Profiler::Report& Graph::get_build_report() const {
    return last_build_report;
}

bool Graph::get_needs_reconnect() const {
    return needs_reconnect;
}

std::ranges::keys_view<std::ranges::ref_view<const std::map<std::string, NodeHandle>>>
Graph::identifiers() {
    return std::as_const(node_for_identifier) | std::ranges::views::keys;
}

ProfilerHandle Graph::prepare_profiler_for_run(InFlightData& in_flight_data) {
    if (!profiler_enable) {
        last_run_report = {};
        return nullptr;
    }

    auto report = run_profiler->set_collect_get_every(
        in_flight_data.profiler_query_pool, profiler_report_intervall_ms, profiler_evict_after_ms);

    if (report) {
        last_run_report = std::move(*report);
        cpu_time_history.set(time_history_current, last_run_report.cpu_total());
        gpu_time_history.set(time_history_current, last_run_report.gpu_total());
        time_history_current++;
    }

    return run_profiler;
}

void Graph::pre_process_connectors(Submission& submission, const NodeHandle& node, NodeData& data) {
    const uint32_t set_idx = data.set_index(run_iteration);
    for (const auto& [input, per_input_info] : data.input_connections) {
        if (!per_input_info.node) {
            continue;
        }
        const GraphResourceHandle& resource =
            std::get<0>(per_input_info.precomputed_resources[set_idx]);
        if ((input->on_pre_process(submission, resource, node, image_barriers, buffer_barriers) &
             Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
            SPDLOG_DEBUG("input {} of {} requested a reconnect",
                         data.input_descriptors.at(input).name, data.label);
            request_reconnect();
        }
    }
    for (const auto& [output, per_output_info] : data.output_connections) {
        const GraphResourceHandle& resource =
            std::get<0>(per_output_info.precomputed_resources[set_idx]);
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
        const GraphResourceHandle& resource =
            std::get<0>(per_input_info.precomputed_resources[set_idx]);
        if ((input->on_post_process(submission, resource, node, image_barriers, buffer_barriers) &
             Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
            SPDLOG_DEBUG("input {} of {} requested a reconnect",
                         data.input_descriptors.at(input).name, data.label);
            request_reconnect();
        }
    }
    for (const auto& [output, per_output_info] : data.output_connections) {
        const GraphResourceHandle& resource =
            std::get<0>(per_output_info.precomputed_resources[set_idx]);
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
