#include "merian/vk/utils/profiler.hpp"

#ifndef MERIAN_PROFILER_ENABLE
#include "spdlog/spdlog.h"
#endif

#define SW_QUERY_COUNT 2

namespace merian {

Profiler::Profiler(const ContextHandle& context)
    : timestamp_period(context->physical_device.get_physical_device_limits().timestampPeriod) {
    cpu_sections.assign(1, {});
    gpu_sections.assign(1, {});
    gpu_sections.reserve(1024);
    cpu_sections.reserve(1024);

#ifndef MERIAN_PROFILER_ENABLE
    SPDLOG_DEBUG("MERIAN_PROFILER_ENABLE not defined. MERIAN_PROFILE_* macros are disabled.");
#endif
}

Profiler::~Profiler() {}

// Remember to reset the query pool after creation
void Profiler::set_query_pool(const QueryPoolHandle<vk::QueryType::eTimestamp>& query_pool) {
    this->query_pool = query_pool;
}

void Profiler::clear() {
    // keep only root
    cpu_sections.assign(1, {});
    gpu_sections.assign(1, {});
    current_gpu_section = 0;
    current_cpu_section = 0;

    clear_index++;
}

void Profiler::cmd_start(const vk::CommandBuffer& cmd,
                         const std::string& name,
                         const vk::PipelineStageFlagBits pipeline_stage) {
    assert(query_pool);
    PerQueryPoolInfo& qp_info = query_pool_infos[query_pool];
    assert(qp_info.pending_gpu_sections.size() * SW_QUERY_COUNT + SW_QUERY_COUNT <
           query_pool->get_query_count());
    assert(query_pool && "num_gpu_timers is 0?");
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
    assert(current_section.timestamp_idx == (uint32_t)-1 &&
           "two sections with the same name or missing collect()?");
    current_section.timestamp_idx = qp_info.pending_gpu_sections.size() * SW_QUERY_COUNT;

    query_pool->write_timestamp(cmd, current_section.timestamp_idx, pipeline_stage);
    qp_info.pending_gpu_sections.emplace_back(current_gpu_section);
}

void Profiler::cmd_end(const vk::CommandBuffer& cmd,
                       const vk::PipelineStageFlagBits pipeline_stage) {
    assert(query_pool && "num_gpu_timers is 0?");
    assert(current_gpu_section != 0 && "missing cmd_start?");
    assert(query_pool);
    GPUSection& section = gpu_sections[current_gpu_section];

    assert(section.timestamp_idx != (uint32_t)-1);
    query_pool->write_timestamp(cmd, section.timestamp_idx + 1, pipeline_stage);
    section.timestamp_idx = (uint32_t)-1;

    current_gpu_section = section.parent_index;
}

void Profiler::collect(const bool wait, const bool keep_query_pool) {
    if (!query_pool) {
        return;
    }

    PerQueryPoolInfo& qp_info = query_pool_infos[query_pool];

    if (qp_info.pending_gpu_sections.empty())
        return;

    assert(current_gpu_section == 0 && "cmd_end missing?");

    if (qp_info.clear_index == clear_index) {
        std::vector<uint64_t> timestamps;
        if (wait) {
            timestamps = query_pool->wait_get_query_pool_results_64(
                0, qp_info.pending_gpu_sections.size() * SW_QUERY_COUNT);
        } else {
            timestamps = query_pool->get_query_pool_results_64(
                0, qp_info.pending_gpu_sections.size() * SW_QUERY_COUNT);
        }

        for (uint32_t i = 0; i < qp_info.pending_gpu_sections.size(); i++) {
            GPUSection& section = gpu_sections[qp_info.pending_gpu_sections[i]];
            section.start = timestamps[SW_QUERY_COUNT * i];
            const uint64_t duration_ns =
                (timestamps[SW_QUERY_COUNT * i + 1] - timestamps[SW_QUERY_COUNT * i]) *
                timestamp_period;
            section.sum_duration_ns += duration_ns;
            section.sq_sum_duration_ns += (duration_ns * duration_ns);
            section.num_captures++;
        }

    } else {
        qp_info.clear_index = clear_index;
    }

    query_pool->reset(0, qp_info.pending_gpu_sections.size() * SW_QUERY_COUNT);

    if (keep_query_pool) {
        qp_info.pending_gpu_sections.clear();
    } else {
        query_pool_infos.erase(query_pool);
    }
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
                      const std::size_t indent_depth = 0) {
    std::string result = "";
    for (const auto& entry : entries) {
        result += fmt::format("{}{}: {:.04f} (± {:.04f}) ms\n", std::string(2 * indent_depth, ' '),
                              entry.name, entry.duration, entry.std_deviation);
        result += to_string(entry.children, indent_depth + 1);
    }
    return result;
}

template <typename TimeMeasure, typename SectionType>
std::vector<Profiler::ReportEntry> make_report(SectionType& section,
                                               std::vector<SectionType>& sections) {
    std::vector<Profiler::ReportEntry> report;
    std::vector<std::pair<TimeMeasure, std::string>> children;
    children.reserve(section.children.size());
    for (auto& child : section.children) {
        children.emplace_back(sections[child.second].start, child.first);
    }
    std::sort(children.begin(), children.end());
    for (auto& child : children) {
        SectionType& subsection = sections[section.children[child.second]];
        if (!subsection.num_captures) {
            continue;
        }

        const double avg = subsection.sum_duration_ns / (double)subsection.num_captures;
        const double std =
            std::sqrt(subsection.sq_sum_duration_ns / (double)subsection.num_captures - avg * avg);

        report.emplace_back(child.second, avg / 1e6, std / 1e6,
                            make_report<TimeMeasure>(subsection, sections));
    }
    return report;
}

std::optional<Profiler::Report>
Profiler::set_collect_get_every(const QueryPoolHandle<vk::QueryType::eTimestamp>& query_pool,
                                const uint32_t report_intervall_millis) {
    set_query_pool(query_pool);

    collect(false, true);

    std::optional<Profiler::Report> report;

    if (report_intervall.millis() >= report_intervall_millis) {
        report = get_report();
        if (gpu_sections.size() > 1 && report.value().gpu_report.empty()) {
            // there are sections but we got no results yet
            return std::nullopt;
        }
        clear();
        report_intervall.reset();
    }

    return report;
}

std::string Profiler::get_report_str(const Profiler::Report& report) {
    if (report.cpu_report.empty() && report.gpu_report.empty()) {
        return "no timestamps captured";
    }

    std::string result = "";
    result += "CPU:\n";
    result += to_string(report.cpu_report, 1);
    result += "GPU:\n";
    result += to_string(report.gpu_report, 1);
    return result;
}

void to_config(Properties& config,
               const std::vector<Profiler::ReportEntry>& entries,
               const uint32_t level = 0) {
    for (const auto& entry : entries) {
        std::string str = fmt::format("{}: {:.04f} (± {:.04f}) ms\n", entry.name, entry.duration,
                                      entry.std_deviation);
        if (entry.children.empty()) {
            // Add 3 spaces to fit with the child item symbol.
            config.output_text(fmt::format("   {}", str.c_str()));
        } else if (config.st_begin_child(fmt::format("{}-{}", level, entry.name), str)) {
            to_config(config, entry.children, level + 1);
            config.st_end_child();
        }
    }
}

void Profiler::get_cpu_report_as_config(Properties& config, const Profiler::Report& report) {
    to_config(config, report.cpu_report);
}

void Profiler::get_gpu_report_as_config(Properties& config, const Profiler::Report& report) {
    to_config(config, report.gpu_report, 1u << 31);
}

void Profiler::get_report_as_config(Properties& config, const Profiler::Report& report) {
    if (!report.cpu_report.empty()) {
        config.st_separate("CPU");
        get_cpu_report_as_config(config, report);
    }

    if (!report.gpu_report.empty()) {
        config.st_separate("GPU");
        get_gpu_report_as_config(config, report);
    }
}

Profiler::Report Profiler::get_report() {
    Profiler::Report report;
    report.cpu_report =
        make_report<chrono_clock::time_point>(cpu_sections[current_cpu_section], cpu_sections);
    report.gpu_report = make_report<uint64_t>(gpu_sections[current_gpu_section], gpu_sections);
    return report;
}

} // namespace merian
