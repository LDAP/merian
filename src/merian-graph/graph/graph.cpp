#include "merian-graph/graph/graph.hpp"
#include "merian-graph/merian_graph_extension.hpp"
#include "merian/shader/glsl_compiler_provider.hpp"
#include "merian/utils/properties.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/extension/extension_registry.hpp"

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
      resource_allocator(create_info.resource_allocator), queue(context->get_queue_GCT()),
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
      low_latency(context), run_profiler(std::make_shared<merian::Profiler>(context)),
      run_info(resource_allocator) {

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

    // A node that presents publishes its swapchain, which is what low latency paces against.
    register_event_listener(
        "//swapchain", [this](const GraphEvent::Info& /*info*/, const GraphEvent::Data& data) {
            if (const SwapchainHandle* const swapchain = std::any_cast<SwapchainHandle>(&data)) {
                low_latency.set_swapchain(*swapchain);
            }
            return false;
        });
}

Graph::~Graph() {
    wait();
}

void Graph::run() {
    // Apply a reload requested from the UI here, before any processing of this iteration.
    if (pending_load) {
        const std::filesystem::path path = *pending_load;
        pending_load.reset();
        wait();
        load_from_file(path);
    }

    // PREPARE RUN: wait for fence, release resources, reset cmd pool
    run_in_progress = true;

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

    low_latency.begin_frame();

    in_flight_data.submission->reset();

    // Compute time stuff
    assert(time_overwrite < TIME_OVERWRITE_COUNT);
    const std::chrono::nanoseconds last_elapsed_ns = duration_elapsed;
    if (time_overwrite == TIME_OVERWRITE_TIME) {
        const auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(time_delta_overwrite_ms / 1000.));
        duration_elapsed += delta;
        duration_elapsed_since_connect += delta;
        time_delta_overwrite_ms = 0;
    } else if (time_overwrite == TIME_OVERWRITE_DELTA) {
        const auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(time_delta_overwrite_ms / 1000.));
        duration_elapsed += delta;
        duration_elapsed_since_connect += delta;
    } else {
        const auto now = std::chrono::high_resolution_clock::now();
        duration_elapsed = now - time_reference;
        duration_elapsed_since_connect = now - time_connect_reference;
    }
    time_delta = duration_elapsed - last_elapsed_ns;

    const ProfilerHandle& profiler = prepare_profiler_for_run(in_flight_data);
    const ScopedDefaultProfiler scoped_default_profiler{profiler};
    const auto run_start = std::chrono::high_resolution_clock::now();

    // CONNECT and PREPROCESS
    do {
        // While connection nodes can signalize that they need to reconnect
        while (needs_reconnect) {
            connect();
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
                    MERIAN_PROFILE_SCOPE(profiler, fmt::format("{} ({})", data.identifier,
                                                               registry.node_type_name(node)));
                    const uint32_t set_idx = data.set_index(run_iteration);
                    Node::NodeStatusFlags flags =
                        node->pre_process(data.resource_maps[set_idx], run_info);
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
    low_latency.begin_render();

    Submission& submission = *in_flight_data.submission;
    {
        MERIAN_PROFILE_SCOPE(profiler, "on_run_starting");
        SPDLOG_TRACE("starting run: iteration: {}", run_info.get_iteration());
        on_run_starting(run_info);
    }
    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, submission.get_cmd(), "Run nodes");
        for (auto& layer : layers) {
            if (layer.barrier.dstStageMask) {
                submission.get_cmd()->barrier(layer.barrier);
            }

            for (auto& node : layer.nodes) {
                NodeData& data = node_data.at(node);

                if (debug_utils) {
                    const std::string node_debug_name =
                        fmt::format("{} ({})", data.identifier, registry.node_type_name(node));
                    debug_utils->cmd_begin_label(*submission.get_cmd(), node_debug_name);
                    SPDLOG_TRACE("running node: {}", node_debug_name);
                }

                run_node(submission, node, data, profiler);

                if (debug_utils)
                    debug_utils->cmd_end_label(*submission.get_cmd());
            }
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
    run_in_progress = false;

    {
        MERIAN_PROFILE_SCOPE(profiler, "on_run_finished_tasks");
        for (const auto& task : on_run_finished_tasks)
            task();
        on_run_finished_tasks.clear();
    }

    cpu_time = std::chrono::high_resolution_clock::now() - run_start;
}

void Graph::wait() {
    SPDLOG_DEBUG("wait until all in-flight iterations have finished");
    ring_fences.wait_all();
    cpu_queue->wait_idle();
}

void Graph::reset() {
    wait();

    node_data.clear();
    node_for_identifier.clear();
    loaded_description = {};
    for (uint32_t i = 0; i < ring_fences.size(); i++) {
        InFlightData& in_flight_data = ring_fences.get(i).user_data;
        in_flight_data.in_flight_data.clear();
    }

    needs_reconnect = true;
}

void Graph::request_reconnect() {
    needs_reconnect = true;
}

void Graph::set_time_delta_overwrite(const float delta_ms) {
    time_overwrite = TIME_OVERWRITE_DELTA;
    time_delta_overwrite_ms = delta_ms;
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

void Graph::run_node(Submission& submission,
                     const NodeHandle& node,
                     NodeData& data,
                     [[maybe_unused]] const ProfilerHandle& profiler) {
    const uint32_t set_idx = data.set_index(run_iteration);

    MERIAN_PROFILE_SCOPE_GPU(
        profiler, submission.get_cmd(),
        fmt::format("{} ({})", data.identifier, registry.node_type_name(node)));

    std::vector<vk::ImageMemoryBarrier2> image_barriers;
    std::vector<vk::BufferMemoryBarrier2> buffer_barriers;

    {
        // Call connector callbacks (pre_process) and record descriptor set updates
        for (auto& [input, per_input_info] : data.input_connections) {
            if (!per_input_info.node) {
                // optional input not connected
                continue;
            }

            auto& [resource, resource_index] = per_input_info.precomputed_resources[set_idx];
            const Connector::ConnectorStatusFlags flags =
                input->on_pre_process(submission, resource, node, image_barriers, buffer_barriers);
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                SPDLOG_DEBUG("input connector {} at node {} requested reconnect.",
                             data.input_name_for_connector.at(input), data.identifier);
                request_reconnect();
            }
        }
        for (auto& [output, per_output_info] : data.output_connections) {
            auto& [resource, resource_index] = per_output_info.precomputed_resources[set_idx];
            const Connector::ConnectorStatusFlags flags =
                output->on_pre_process(submission, resource, node, image_barriers, buffer_barriers);
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                SPDLOG_DEBUG("output connector {} at node {} requested reconnect.",
                             data.output_name_for_connector.at(output), data.identifier);
                request_reconnect();
            }
        }

        if (!image_barriers.empty()) {
            submission.get_cmd()->barrier(image_barriers);
            image_barriers.clear();
        }
        if (!buffer_barriers.empty()) {
            submission.get_cmd()->barrier(buffer_barriers);
            buffer_barriers.clear();
        }
    }

    {
        try {
            const Node::NodeStatusFlags flags =
                node->process(data.resource_maps[set_idx], run_info, submission);
            if ((flags & Node::NodeStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                SPDLOG_DEBUG("node {} requested reconnect in process", data.identifier);
                request_reconnect();
            }
            if ((flags & Node::NodeStatusFlagBits::REMOVE_NODE) != 0u) {
                remove_node(data.identifier);
            }
        } catch (const graph_errors::node_error& e) {
            data.errors_queued.emplace_back(fmt::format("node error: {}", e.what()));
        } catch (const GLSLShaderCompiler::compilation_failed& e) {
            data.errors_queued.emplace_back(fmt::format("compilation failed: {}", e.what()));
        }
        if (!data.errors_queued.empty()) {
            SPDLOG_ERROR("executing node '{}' failed:\n - {}", data.identifier,
                         fmt::join(data.errors_queued, "\n   - "));
            request_reconnect();
            SPDLOG_ERROR("emergency reconnect.");
        }
    }

    {
        // Call connector callbacks (post_process) and record descriptor set updates
        for (auto& [input, per_input_info] : data.input_connections) {
            if (!per_input_info.node) {
                // optional input not connected
                continue;
            }

            auto& [resource, resource_index] = per_input_info.precomputed_resources[set_idx];
            const Connector::ConnectorStatusFlags flags =
                input->on_post_process(submission, resource, node, image_barriers, buffer_barriers);
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                SPDLOG_DEBUG("input connector {} at node {} requested reconnect.",
                             data.input_name_for_connector.at(input), data.identifier);
                request_reconnect();
            }
        }
        for (auto& [output, per_output_info] : data.output_connections) {
            auto& [resource, resource_index] = per_output_info.precomputed_resources[set_idx];
            const Connector::ConnectorStatusFlags flags = output->on_post_process(
                submission, resource, node, image_barriers, buffer_barriers);
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                SPDLOG_DEBUG("output connector {} at node {} requested reconnect.",
                             data.output_name_for_connector.at(output), data.identifier);
                request_reconnect();
            }
        }

        if (!image_barriers.empty()) {
            submission.get_cmd()->barrier(image_barriers);
        }
        if (!buffer_barriers.empty()) {
            submission.get_cmd()->barrier(buffer_barriers);
        }
    }
}

} // namespace merian

REGISTER_CONTEXT_EXTENSION(merian::MerianGraphExtension, "merian-graph");
