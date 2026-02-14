#include "merian-nodes/graph/graph.hpp"
#include "merian-nodes/merian_nodes_extension.hpp"
#include "merian/vk/extension/extension_glsl_compiler.hpp"
#include "merian/vk/extension/extension_registry.hpp"

#include <spdlog/spdlog.h>

namespace merian {

using namespace merian;
using namespace graph_internal;
using namespace std::literals::chrono_literals;

const ContextHandle& check_requirements_and_get_context(const ContextHandle& context) {
    if (context->get_context_extension<MerianNodesExtension>(true) == nullptr) {
        throw graph_errors::graph_error{
            "The merian-nodes context extension must be enabled and supported."};
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
                      in_flight_data.command_pool = std::make_shared<CommandPool>(queue);
                      in_flight_data.command_buffer_cache =
                          std::make_shared<CachingCommandPool>(in_flight_data.command_pool);
                      in_flight_data.profiler_query_pool =
                          std::make_shared<merian::QueryPool<vk::QueryType::eTimestamp>>(context,
                                                                                         512, true);
                      return in_flight_data;
                  }),
      run_profiler(std::make_shared<merian::Profiler>(context)),
      graph_run(thread_pool,
                cpu_queue,
                resource_allocator,
                queue,
                context->get_context_extension<ExtensionGLSLCompiler>()->get_compiler()) {

    debug_utils = context->get_context_extension<ExtensionVkDebugUtils>(true);
    time_connect_reference = time_reference = std::chrono::high_resolution_clock::now();
    duration_elapsed = 0ns;
    context_extension = context->get_context_extension<MerianNodesExtension>();
}

Graph::~Graph() {
    wait();
}

void Graph::run() {
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

    // LOW LATENCY MODE
    if (low_latency_mode && !needs_reconnect) {
        const auto total_wait = std::max(
            (std::max(gpu_wait_time, external_wait_time) + in_flight_data.cpu_sleep_time - 0.1ms),
            0.00ms);
        in_flight_data.cpu_sleep_time = 0.92 * total_wait;
    } else {
        in_flight_data.cpu_sleep_time = 0ms;
    }

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

    const std::shared_ptr<CachingCommandPool>& cmd_cache = in_flight_data.command_buffer_cache;
    cmd_cache->reset();

    // Compute time stuff
    assert(time_overwrite < 3);
    const std::chrono::nanoseconds last_elapsed_ns = duration_elapsed;
    if (time_overwrite == 1) {
        const auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(time_delta_overwrite_ms / 1000.));
        duration_elapsed += delta;
        duration_elapsed_since_connect += delta;
        time_delta_overwrite_ms = 0;
    } else if (time_overwrite == 2) {
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
    const auto run_start = std::chrono::high_resolution_clock::now();

    // CONNECT and PREPROCESS
    do {
        // While connection nodes can signalize that they need to reconnect
        while (needs_reconnect) {
            connect();
        }

        graph_run.begin_run(ring_fences.size(), cmd_cache, run_iteration, total_iteration,
                            ring_fences.current_cycle_index(), time_delta, duration_elapsed,
                            duration_elapsed_since_connect, profiler);

        // While preprocessing nodes can signalize that they need to reconnect as well
        {
            MERIAN_PROFILE_SCOPE(profiler, "Preprocess nodes");
            for (auto& node : flat_topology) {
                NodeData& data = node_data.at(node);
                MERIAN_PROFILE_SCOPE(profiler, fmt::format("{} ({})", data.identifier,
                                                           registry.node_type_name(node)));
                const uint32_t set_idx = data.set_index(run_iteration);
                Node::NodeStatusFlags flags =
                    node->pre_process(graph_run, data.resource_maps[set_idx]);
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
    {
        MERIAN_PROFILE_SCOPE(profiler, "on_run_starting");
        SPDLOG_TRACE("starting run: iteration: {}", graph_run.get_iteration());
        on_run_starting(graph_run);
    }
    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, graph_run.get_cmd(), "Run nodes");
        for (auto& node : flat_topology) {
            NodeData& data = node_data.at(node);
            if (debug_utils) {
                const std::string node_debug_name =
                    fmt::format("{} ({})", data.identifier, registry.node_type_name(node));
                debug_utils->cmd_begin_label(*graph_run.get_cmd(), node_debug_name);
                SPDLOG_TRACE("running node: {}", node_debug_name);
            }

            try {
                run_node(graph_run, node, data, profiler);
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

            if (debug_utils)
                debug_utils->cmd_end_label(*graph_run.get_cmd());
        }
    }

    // FINISH RUN: submit

    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, graph_run.get_cmd(), "on_pre_submit");
        on_pre_submit(graph_run);
    }

    {

        MERIAN_PROFILE_SCOPE(profiler, "end run");
        graph_run.end_run(ring_fences.reset());
    }
    {
        MERIAN_PROFILE_SCOPE(profiler, "on_post_submit");
        on_post_submit();
    }

    external_wait_time = 0.9 * external_wait_time + 0.1 * graph_run.external_wait_time;
    needs_reconnect |= graph_run.needs_reconnect;
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
    for (uint32_t i = 0; i < ring_fences.size(); i++) {
        InFlightData& in_flight_data = ring_fences.get(i).user_data;
        in_flight_data.in_flight_data.clear();
    }

    needs_reconnect = true;
}

void Graph::request_reconnect() {
    needs_reconnect = true;
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

    auto report = run_profiler->set_collect_get_every(in_flight_data.profiler_query_pool,
                                                      profiler_report_intervall_ms);

    if (report) {
        last_run_report = std::move(*report);
        cpu_time_history.set(time_history_current, last_run_report.cpu_total());
        gpu_time_history.set(time_history_current, last_run_report.gpu_total());
        time_history_current++;
    }

    return run_profiler;
}

void Graph::run_node(GraphRun& run,
                     const NodeHandle& node,
                     NodeData& data,
                     [[maybe_unused]] const ProfilerHandle& profiler) {
    const uint32_t set_idx = data.set_index(run_iteration);

    MERIAN_PROFILE_SCOPE_GPU(
        profiler, run.get_cmd(),
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
            const Connector::ConnectorStatusFlags flags = input->on_pre_process(
                run, run.get_cmd(), resource, node, image_barriers, buffer_barriers);
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) != 0u) {
                NodeData& src_data = node_data.at(per_input_info.node);
                record_descriptor_updates(src_data, per_input_info.output,
                                          src_data.output_connections[per_input_info.output],
                                          resource_index);
            }
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                SPDLOG_DEBUG("input connector {} at node {} requested reconnect.",
                             data.input_name_for_connector.at(input), data.identifier);
                request_reconnect();
            }
        }
        for (auto& [output, per_output_info] : data.output_connections) {
            auto& [resource, resource_index] = per_output_info.precomputed_resources[set_idx];
            const Connector::ConnectorStatusFlags flags = output->on_pre_process(
                run, run.get_cmd(), resource, node, image_barriers, buffer_barriers);
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) != 0u) {
                record_descriptor_updates(data, output, per_output_info, resource_index);
            }
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                SPDLOG_DEBUG("output connector {} at node {} requested reconnect.",
                             data.output_name_for_connector.at(output), data.identifier);
                request_reconnect();
            }
        }

        if (!image_barriers.empty() || !buffer_barriers.empty()) {
            vk::DependencyInfoKHR dep_info{{}, {}, buffer_barriers, image_barriers};
            run.get_cmd()->barrier({}, buffer_barriers, image_barriers);
            image_barriers.clear();
            buffer_barriers.clear();
        }
    }

    auto& descriptor_set = data.descriptor_sets[set_idx];
    {
        // apply descriptor set updates
        data.statistics.last_descriptor_set_updates = descriptor_set->update_count();
        if (descriptor_set->has_updates()) {
            SPDLOG_TRACE("applying {} descriptor set updates for node {}, set {}",
                         descriptor_set->update_count(), data.identifier, set_idx);
            descriptor_set->update();
        }
    }

    {
        node->process(run, descriptor_set, data.resource_maps[set_idx]);
#ifndef NDEBUG
        if (run.needs_reconnect && !get_needs_reconnect()) {
            SPDLOG_DEBUG("node {} requested reconnect in process", data.identifier);
            request_reconnect();
        }
#endif
    }

    {
        // Call connector callbacks (post_process) and record descriptor set updates
        for (auto& [input, per_input_info] : data.input_connections) {
            if (!per_input_info.node) {
                // optional input not connected
                continue;
            }

            auto& [resource, resource_index] = per_input_info.precomputed_resources[set_idx];
            const Connector::ConnectorStatusFlags flags = input->on_post_process(
                run, run.get_cmd(), resource, node, image_barriers, buffer_barriers);
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) != 0u) {
                NodeData& src_data = node_data.at(per_input_info.node);
                record_descriptor_updates(src_data, per_input_info.output,
                                          src_data.output_connections[per_input_info.output],
                                          resource_index);
            }
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                SPDLOG_DEBUG("input connector {} at node {} requested reconnect.",
                             data.input_name_for_connector.at(input), data.identifier);
                request_reconnect();
            }
        }
        for (auto& [output, per_output_info] : data.output_connections) {
            auto& [resource, resource_index] = per_output_info.precomputed_resources[set_idx];
            const Connector::ConnectorStatusFlags flags = output->on_post_process(
                run, run.get_cmd(), resource, node, image_barriers, buffer_barriers);
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_DESCRIPTOR_UPDATE) != 0u) {
                record_descriptor_updates(data, output, per_output_info, resource_index);
            }
            if ((flags & Connector::ConnectorStatusFlagBits::NEEDS_RECONNECT) != 0u) {
                SPDLOG_DEBUG("output connector {} at node {} requested reconnect.",
                             data.output_name_for_connector.at(output), data.identifier);
                request_reconnect();
            }
        }

        if (!image_barriers.empty() || !buffer_barriers.empty()) {
            run.get_cmd()->barrier({}, buffer_barriers, image_barriers);
        }
    }
}

void Graph::record_descriptor_updates(NodeData& src_data,
                                      const OutputConnectorHandle& src_output,
                                      NodeData::PerOutputInfo& per_output_info,
                                      const uint32_t resource_index) {
    NodeData::PerResourceInfo& resource_info = per_output_info.resources[resource_index];

    if (per_output_info.descriptor_set_binding != DescriptorSet::NO_DESCRIPTOR_BINDING)
        for (auto& set_idx : resource_info.set_indices) {
            src_output->get_descriptor_update(
                per_output_info.descriptor_set_binding, resource_info.resource,
                src_data.descriptor_sets[set_idx], resource_allocator);
        }

    for (auto& [dst_node, dst_input, set_idx] : resource_info.other_set_indices) {
        NodeData& dst_data = node_data.at(dst_node);
        NodeData::PerInputInfo& per_input_info = dst_data.input_connections[dst_input];
        if (per_input_info.descriptor_set_binding != DescriptorSet::NO_DESCRIPTOR_BINDING)
            dst_input->get_descriptor_update(per_input_info.descriptor_set_binding,
                                             resource_info.resource,
                                             dst_data.descriptor_sets[set_idx], resource_allocator);
    }
}

} // namespace merian

REGISTER_CONTEXT_EXTENSION(merian::MerianNodesExtension, "merian-nodes");
