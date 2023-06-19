#include "merian/vk/utils/profiler.hpp"
#include "merian/vk/utils/check_result.hpp"
#include <queue>

#define SW_QUERY_COUNT 2

namespace merian {

Profiler::Profiler(const SharedContext context, const uint32_t num_gpu_timers)
    : context(context), num_gpu_timers(num_gpu_timers) {
    vk::QueryPoolCreateInfo createInfo({}, vk::QueryType::eTimestamp,
                                       num_gpu_timers * SW_QUERY_COUNT);
    query_pool = context->device.createQueryPool(createInfo);
    pending_gpu_timestamps.reserve(num_gpu_timers * SW_QUERY_COUNT);
    gpu_sections.resize(num_gpu_timers);
    cpu_sections.resize(1024);

    timestamp_period =
        context->pd_container.physical_device_props.properties.limits.timestampPeriod;
}

Profiler::~Profiler() {
    context->device.waitIdle();
    context->device.destroyQueryPool(query_pool);
}

void Profiler::cmd_reset(const vk::CommandBuffer& cmd) {
    cmd.resetQueryPool(query_pool, 0, num_gpu_timers);
    cpu_sections.clear();
    gpu_sections.clear();
    pending_gpu_timestamps.clear();
    reset_was_called = true;
}

uint32_t Profiler::cmd_start(const vk::CommandBuffer& cmd,
                             const std::string name,
                             const vk::PipelineStageFlagBits pipeline_stage) {
    assert(reset_was_called);
    assert(pending_gpu_timestamps.size() < num_gpu_timers);

    GPUSection section;
    section.name = name;
    section.start_timestamp_idx = pending_gpu_timestamps.size();
    cmd.writeTimestamp(pipeline_stage, query_pool, section.start_timestamp_idx);
    const uint32_t section_index = gpu_sections.size();
    gpu_sections.push_back(section);
    pending_gpu_timestamps.emplace_back(section_index, false);
    return section_index;
}

void Profiler::cmd_end(const vk::CommandBuffer& cmd,
                       const uint32_t start_id,
                       const vk::PipelineStageFlagBits pipeline_stage) {
    assert(reset_was_called);
    assert(start_id < gpu_sections.size());
    assert(pending_gpu_timestamps.size() < num_gpu_timers);

    GPUSection& section = gpu_sections[start_id];
    section.end_timestamp_idx = pending_gpu_timestamps.size();
    cmd.writeTimestamp(pipeline_stage, query_pool, section.end_timestamp_idx);
    pending_gpu_timestamps.emplace_back(start_id, true);
}

void Profiler::collect(const bool wait) {
    if (pending_gpu_timestamps.empty())
        return;

    assert(reset_was_called);

    vk::QueryResultFlags flags = vk::QueryResultFlagBits::e64;
    if (wait) {
        flags |= vk::QueryResultFlagBits::eWait;
    }

    std::vector<uint64_t> timestamps(pending_gpu_timestamps.size());
    check_result(
        context->device.getQueryPoolResults(query_pool, 0, pending_gpu_timestamps.size(),
                                            sizeof(uint64_t) * pending_gpu_timestamps.size(),
                                            timestamps.data(), sizeof(uint64_t), flags),
        "could not get query results");

    for (uint32_t i = 0; i < pending_gpu_timestamps.size(); i++) {
        auto& [gpu_sec_idx, is_end] = pending_gpu_timestamps[i];
        uint64_t ts = timestamps[i];
        GPUSection& section = gpu_sections[gpu_sec_idx];
        if (is_end) {
            section.end = ts;
        } else {
            section.start = ts;
        }
        if (section.start && section.end) {
            section.duration_ns = (section.end - section.start) * timestamp_period;
        }
    }

    reset_was_called = false;
}

uint32_t Profiler::start(const std::string& name) {
    CPUSection section;
    section.name = name;
    section.start = chrono_clock::now();
    uint32_t section_index = cpu_sections.size();
    cpu_sections.push_back(section);

    std::atomic_signal_fence(std::memory_order_seq_cst);
    return section_index;
}

void Profiler::end(const uint32_t start_id) {
    assert(start_id < cpu_sections.size());
    std::atomic_signal_fence(std::memory_order_seq_cst);

    CPUSection& section = cpu_sections[start_id];
    section.end = chrono_clock::now();
    section.duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(section.end - section.start).count();
}

std::string Profiler::get_report() {
    std::string result = "";

    // timestamp, is_end, section index
    std::priority_queue<std::tuple<chrono_clock::time_point, bool, uint32_t>,
                        std::vector<std::tuple<chrono_clock::time_point, bool, uint32_t>>,
                        std::greater<std::tuple<chrono_clock::time_point, bool, uint32_t>>>
        q;

    result += "CPU:\n\n";
    for (uint32_t i = 0; i < cpu_sections.size(); i++) {
        CPUSection& section = cpu_sections[i];
        q.emplace(section.start, false, i);
        q.emplace(section.end, true, i);
    }

    std::string indent = "";
    while (!q.empty()) {
        auto& [ts, is_end, section_index] = q.top();
        if (!is_end) {
            CPUSection& section = cpu_sections[section_index];
            result += fmt::format("{}{}: {} ms\n", indent, section.name,
                                  section.duration_ns / (double)1e6);
        }
        if (is_end) {
            indent = indent.substr(0, std::max(0, (int)indent.size() - 2));
        } else {
            indent += "  ";
        }
        q.pop();
    }

    std::priority_queue<std::tuple<uint64_t, bool, uint32_t>,
                        std::vector<std::tuple<uint64_t, bool, uint32_t>>,
                        std::greater<std::tuple<uint64_t, bool, uint32_t>>>
        gpu_q;
    result += "\n\nGPU:\n\n";
    for (uint32_t i = 0; i < gpu_sections.size(); i++) {
        GPUSection& section = gpu_sections[i];
        gpu_q.emplace(section.start, false, i);
        gpu_q.emplace(section.end, true, i);
    }

    indent = "";
    while (!gpu_q.empty()) {
        auto& [ts, is_end, section_index] = gpu_q.top();
        if (!is_end) {
            GPUSection& section = gpu_sections[section_index];
            result += fmt::format("{}{}: {} ms\n", indent, section.name,
                                  section.duration_ns / (double)1e6);
        }
        if (is_end) {
            indent = indent.substr(0, std::max(0, (int)indent.size() - 2));
        } else {
            indent += "  ";
        }
        gpu_q.pop();
    }

    return result;
}

} // namespace merian
