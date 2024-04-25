#include "merian/vk/utils/profiler.hpp"
#include "imgui.h"
#include "merian/vk/utils/check_result.hpp"
#include <queue>

#define SW_QUERY_COUNT 2

namespace merian {

Profiler::Profiler(const SharedContext context, const QueueHandle queue, const uint32_t num_gpu_timers)
    : context(context), num_gpu_timers(num_gpu_timers) {
    vk::QueryPoolCreateInfo createInfo({}, vk::QueryType::eTimestamp,
                                       num_gpu_timers * SW_QUERY_COUNT);
    query_pool = context->device.createQueryPool(createInfo);
    pending_gpu_timestamps.reserve(num_gpu_timers * SW_QUERY_COUNT);
    gpu_sections.reserve(num_gpu_timers);
    cpu_sections.reserve(1024);

    const uint64_t valid_bits = context->physical_device.physical_device.getQueueFamilyProperties()[queue->get_queue_family_index()].timestampValidBits;
    if (valid_bits < 64) {
        bitmask = (((uint64_t)1) << (valid_bits + 1)) - 1;
    } else {
        bitmask = (uint64_t)-1;
    }
    SPDLOG_DEBUG("using queue with valid bits: {}, mask: {}", valid_bits, bitmask);

    timestamp_period =
        context->physical_device.physical_device_properties.properties.limits.timestampPeriod;
}

Profiler::~Profiler() {
    context->device.waitIdle();
    context->device.destroyQueryPool(query_pool);
}

void Profiler::cmd_reset(const vk::CommandBuffer& cmd, const bool clear) {
    cmd.resetQueryPool(query_pool, 0, num_gpu_timers);
    pending_gpu_timestamps.clear();
    reset_was_called = true;

    if (clear) {
        cpu_sections.clear();
        gpu_sections.clear();
        cpu_key_to_section_idx.clear();
        gpu_key_to_section_idx.clear();
    }
}

uint32_t Profiler::cmd_start(const vk::CommandBuffer& cmd,
                             const std::string name,
                             const vk::PipelineStageFlagBits pipeline_stage) {
    assert(reset_was_called);
    assert(pending_gpu_timestamps.size() < num_gpu_timers);

    gpu_current_key += "$$" + name;
    uint32_t section_index;
    if (gpu_key_to_section_idx.contains(gpu_current_key)) {
        section_index = gpu_key_to_section_idx[gpu_current_key];
    } else {
        section_index = gpu_sections.size();
        gpu_key_to_section_idx[gpu_current_key] = section_index;
        gpu_sections.emplace_back();
        gpu_sections.back().name = name;
    }
    gpu_sections[section_index].start = 0;
    gpu_sections[section_index].end = 0;
    gpu_sections[section_index].start_timestamp_idx = pending_gpu_timestamps.size();
    cmd.writeTimestamp(pipeline_stage, query_pool, gpu_sections[section_index].start_timestamp_idx);
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

    gpu_current_key.resize(gpu_current_key.size() - 2 - section.name.size());
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
        uint64_t ts = timestamps[i] & bitmask;
        GPUSection& section = gpu_sections[gpu_sec_idx];
        if (is_end) {
            section.end = ts;
        } else {
            section.start = ts;
        }
        if (section.start && section.end) {
            const uint64_t duration_ns = (section.end - section.start) * timestamp_period;
            section.sum_duration_ns += duration_ns;
            section.sq_sum_duration_ns += (duration_ns * duration_ns);
            section.num_captures++;
        }
    }

    reset_was_called = false;
}

uint32_t Profiler::start(const std::string& name) {
    cpu_current_key += "$$" + name;
    uint32_t section_index;
    if (cpu_key_to_section_idx.contains(cpu_current_key)) {
        section_index = cpu_key_to_section_idx[cpu_current_key];
    } else {
        section_index = cpu_sections.size();
        cpu_key_to_section_idx[cpu_current_key] = section_index;
        cpu_sections.emplace_back();
        cpu_sections.back().name = name;
    }

    cpu_sections[section_index].start = chrono_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);

    return section_index;
}

void Profiler::end(const uint32_t start_id) {
    assert(start_id < cpu_sections.size());
    std::atomic_signal_fence(std::memory_order_seq_cst);

    CPUSection& section = cpu_sections[start_id];
    section.end = chrono_clock::now();
    uint64_t duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(section.end - section.start).count();
    section.sum_duration_ns += duration_ns;
    section.sq_sum_duration_ns += (duration_ns * duration_ns);
    section.num_captures++;
    cpu_current_key.resize(cpu_current_key.size() - 2 - section.name.size());
}

std::string to_string(const std::vector<Profiler::ReportEntry>& entries,
                      const std::string indent = "") {
    std::string result = "";
    for (auto& entry : entries) {
        result += fmt::format("{}{}: {:.04f} (± {:.04f}) ms\n", indent, entry.name, entry.duration,
                              entry.std_deviation);
        result += to_string(entry.children, indent + "  ");
    }
    return result;
}

template <typename TimeMeasure, typename SectionType>
std::vector<Profiler::ReportEntry>
make_report(std::vector<SectionType>& sections,
            std::priority_queue<std::tuple<TimeMeasure, bool, uint32_t>,
                                std::vector<std::tuple<TimeMeasure, bool, uint32_t>>,
                                std::greater<std::tuple<TimeMeasure, bool, uint32_t>>>& queue) {
    std::vector<Profiler::ReportEntry> report;
    bool last_was_start = false;
    while (!queue.empty()) {
        auto& [ts, is_end, section_index] = queue.top();
        if (!last_was_start && !is_end) {
            const SectionType& section = sections[section_index];
            const double avg = section.sum_duration_ns / (double)section.num_captures;
            const double std =
                std::sqrt(section.sq_sum_duration_ns / (double)section.num_captures - avg * avg);

            report.emplace_back(section.name, avg / 1e6, std / 1e6,
                                std::vector<Profiler::ReportEntry>());
            last_was_start = true;
            queue.pop();
            continue;
        }
        if (last_was_start && !is_end) {
            report.back().children = make_report(sections, queue);
            last_was_start = true;
            continue;
        }
        if (!last_was_start && is_end) {
            return report;
        }
        if (last_was_start && is_end) {
            queue.pop();
            last_was_start = false;
            continue;
        }
    }
    return report;
}

std::string Profiler::get_report_str() {
    Profiler::Report report = get_report();

    if (report.cpu_report.empty() && report.gpu_report.empty()) {
        return "no timestamps captured";
    }

    std::string result = "";
    result += "CPU:\n";
    result += to_string(report.cpu_report);
    result += "GPU:\n";
    result += to_string(report.gpu_report);
    return result;
}

void to_imgui(const std::vector<Profiler::ReportEntry>& entries, const uint32_t level = 0) {
    for (auto& entry : entries) {
        std::string str = fmt::format("{}: {:.04f} (± {:.04f}) ms\n", entry.name, entry.duration,
                                      entry.std_deviation);
        if (entry.children.empty()) {
            // Add 3 spaces to fit with the child item symbol.
            ImGui::Text("   %s", str.c_str());
        } else if (ImGui::TreeNode(fmt::format("{}-{}", level, entry.name).c_str(), "%s",
                                   str.c_str())) {
            to_imgui(entry.children, level + 1);
            ImGui::TreePop();
        }
    }
}

void Profiler::get_report_imgui() {
    get_report_imgui(get_report());
}

void Profiler::get_report_imgui(const Profiler::Report& report) {
    if (ImGui::CollapsingHeader("Profiler")) {
        ImGui::SeparatorText("CPU");
        if (report.cpu_report.empty())
            ImGui::Text("nothing captured");
        else
            to_imgui(report.cpu_report);

        ImGui::SeparatorText("GPU");
        if (report.cpu_report.empty())
            ImGui::Text("nothing captured");
        else
            to_imgui(report.gpu_report, 1u << 31);
    }
}

Profiler::Report Profiler::get_report() {
    Profiler::Report report;

    // timestamp, is_end, section index
    std::priority_queue<std::tuple<chrono_clock::time_point, bool, uint32_t>,
                        std::vector<std::tuple<chrono_clock::time_point, bool, uint32_t>>,
                        std::greater<std::tuple<chrono_clock::time_point, bool, uint32_t>>>
        cpu_queue;
    for (uint32_t i = 0; i < cpu_sections.size(); i++) {
        CPUSection& section = cpu_sections[i];
        cpu_queue.emplace(section.start, false, i);
        cpu_queue.emplace(section.end, true, i);
    }
    report.cpu_report = make_report(cpu_sections, cpu_queue);

    std::priority_queue<std::tuple<uint64_t, bool, uint32_t>,
                        std::vector<std::tuple<uint64_t, bool, uint32_t>>,
                        std::greater<std::tuple<uint64_t, bool, uint32_t>>>
        gpu_queue;
    for (uint32_t i = 0; i < gpu_sections.size(); i++) {
        GPUSection& section = gpu_sections[i];
        gpu_queue.emplace(section.start, false, i);
        gpu_queue.emplace(section.end, true, i);
    }
    report.gpu_report = make_report(gpu_sections, gpu_queue);
    return report;
}

} // namespace merian
