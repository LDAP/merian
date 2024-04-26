#include "merian/vk/utils/profiler.hpp"
#include "imgui.h"
#include "merian/vk/utils/check_result.hpp"

#define SW_QUERY_COUNT 2

namespace merian {

Profiler::Profiler(const SharedContext context,
                   const QueueHandle queue,
                   const uint32_t num_gpu_timers)
    : context(context), num_gpu_timers(num_gpu_timers),
      timestamp_period(context->physical_device.get_physical_device_limits().timestampPeriod) {

    // No special handling necessary. According to the spec "Bits outside the valid range are
    // guaranteed to be zeros."
    const uint64_t valid_bits = queue->get_queue_family_properties().timestampValidBits;
    if (!timestamp_period || !valid_bits) {
        throw std::runtime_error{"device does not support timestamp queries!"};
    }
    SPDLOG_DEBUG("using queue with valid bits: {}. Timestamp period {}", valid_bits,
                 timestamp_period);

    vk::QueryPoolCreateInfo createInfo({}, vk::QueryType::eTimestamp,
                                       num_gpu_timers * SW_QUERY_COUNT);
    query_pool = context->device.createQueryPool(createInfo);
    pending_gpu_sections.reserve(num_gpu_timers * SW_QUERY_COUNT);
    cpu_sections.assign(1, {});
    gpu_sections.assign(1, {});
    gpu_sections.reserve(num_gpu_timers);
    cpu_sections.reserve(1024);

#ifndef MERIAN_PROFILER_ENABLE
    SPDLOG_DEBUG("MERIAN_PROFILER_ENABLE not defined. MERIAN_PROFILE_* macros are disabled.");
#endif
}

Profiler::~Profiler() {
    context->device.waitIdle();
    context->device.destroyQueryPool(query_pool);
}

void Profiler::cmd_reset(const vk::CommandBuffer& cmd, const bool clear) {
    cmd.resetQueryPool(query_pool, 0, num_gpu_timers);
    pending_gpu_sections.clear();
    reset_was_called = true;

    assert(clear || (current_gpu_section == 0 && current_cpu_section == 0 &&
                     "it seams that there is a *end missing?"));
    if (clear) {
        // keep only root
        cpu_sections.assign(1, {});
        gpu_sections.assign(1, {});
        current_gpu_section = 0;
        current_cpu_section = 0;
    }
}

void Profiler::cmd_start(const vk::CommandBuffer& cmd,
                         const std::string name,
                         const vk::PipelineStageFlagBits pipeline_stage) {
    assert(reset_was_called);
    assert(pending_gpu_sections.size() * SW_QUERY_COUNT < num_gpu_timers);

    GPUSection& parent_section = gpu_sections[current_gpu_section];
    if (parent_section.children.contains(name)) {
        current_gpu_section = parent_section.children[name];
    } else {
        gpu_sections.emplace_back();
        gpu_sections.back().parent_index = current_gpu_section;
        current_gpu_section = gpu_sections.size() - 1;
        parent_section.children[name] = current_gpu_section;
    }
    GPUSection& current_section = gpu_sections[current_gpu_section];
    assert(current_section.timestamp_idx == (uint32_t)-1);
    current_section.timestamp_idx = pending_gpu_sections.size() * SW_QUERY_COUNT;

    cmd.writeTimestamp(pipeline_stage, query_pool, current_section.timestamp_idx);
    pending_gpu_sections.emplace_back(current_gpu_section);
}

void Profiler::cmd_end(const vk::CommandBuffer& cmd,
                       const vk::PipelineStageFlagBits pipeline_stage) {
    assert(current_gpu_section != 0 && "missing cmd_start?");
    assert(reset_was_called);
    assert(pending_gpu_sections.size() * SW_QUERY_COUNT < num_gpu_timers);

    GPUSection& section = gpu_sections[current_gpu_section];
    assert(section.timestamp_idx != (uint32_t)-1);
    cmd.writeTimestamp(pipeline_stage, query_pool, section.timestamp_idx + 1);

    current_gpu_section = section.parent_index;
}

void Profiler::collect(const bool wait) {
    if (pending_gpu_sections.empty())
        return;

    assert(reset_was_called);
    assert(current_gpu_section == 0 && "cmd_end missing?");

    vk::QueryResultFlags flags = vk::QueryResultFlagBits::e64;
    if (wait) {
        flags |= vk::QueryResultFlagBits::eWait;
    }

    std::vector<uint64_t> timestamps(pending_gpu_sections.size() * SW_QUERY_COUNT);
    check_result(context->device.getQueryPoolResults(query_pool, 0, timestamps.size(),
                                                     sizeof(uint64_t) * timestamps.size(),
                                                     timestamps.data(), sizeof(uint64_t), flags),
                 "could not get query results");
    for (auto section_index : pending_gpu_sections) {
        GPUSection& section = gpu_sections[section_index];
        section.start = timestamps[section.timestamp_idx];
        const uint64_t duration_ns =
            (timestamps[section.timestamp_idx + 1] - timestamps[section.timestamp_idx]) *
            timestamp_period;
        section.timestamp_idx = (uint32_t)-1;
        section.sum_duration_ns += duration_ns;
        section.sq_sum_duration_ns += (duration_ns * duration_ns);
        section.num_captures++;
    }

    reset_was_called = false;
}

void Profiler::start(const std::string& name) {
    CPUSection& parent_section = cpu_sections[current_cpu_section];
    if (parent_section.children.contains(name)) {
        current_cpu_section = parent_section.children[name];
    } else {
        cpu_sections.emplace_back();
        cpu_sections.back().parent_index = current_cpu_section;
        current_cpu_section = cpu_sections.size() - 1;
        parent_section.children[name] = current_cpu_section;
    }

    cpu_sections[current_cpu_section].start = chrono_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

void Profiler::end() {
    assert(current_cpu_section != 0 && "missing start?");
    std::atomic_signal_fence(std::memory_order_seq_cst);

    CPUSection& section = cpu_sections[current_cpu_section];
    section.end = chrono_clock::now();
    uint64_t duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(section.end - section.start).count();
    section.sum_duration_ns += duration_ns;
    section.sq_sum_duration_ns += (duration_ns * duration_ns);
    section.num_captures++;

    current_cpu_section = section.parent_index;
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
std::vector<Profiler::ReportEntry> make_report(SectionType& section,
                                               std::vector<SectionType>& sections) {
    std::vector<Profiler::ReportEntry> report;
    std::vector<std::pair<TimeMeasure, std::string>> children;
    for (auto& child : section.children) {
        children.emplace_back(sections[child.second].start, child.first);
    }
    std::sort(children.begin(), children.end());
    for (auto& child : children) {
        SectionType& subsection = sections[section.children[child.second]];
        const double avg = subsection.sum_duration_ns / (double)subsection.num_captures;
        const double std =
            std::sqrt(subsection.sq_sum_duration_ns / (double)subsection.num_captures - avg * avg);

        report.emplace_back(child.second, avg / 1e6, std / 1e6,
                            make_report<TimeMeasure>(subsection, sections));
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
    report.cpu_report = make_report<chrono_clock::time_point>(cpu_sections.front(), cpu_sections);
    report.gpu_report = make_report<uint64_t>(gpu_sections.front(), gpu_sections);
    return report;
}

} // namespace merian
