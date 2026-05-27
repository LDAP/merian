#include "merian/vk/utils/profiler.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/command/command_buffer.hpp"

#include "spdlog/spdlog.h"

#include <mutex>
#include <vector>

#define SW_QUERY_COUNT 2

namespace merian {

Profiler::Profiler(const ContextHandle& context)
    : timestamp_period(context->get_physical_device()->get_device_limits().timestampPeriod),
      last_clear_time(chrono_clock::now()) {
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

template <typename SectionType>
void Profiler::enter_child(std::vector<SectionType>& sections,
                           const std::size_t parent_idx,
                           const std::string& name,
                           uint32_t& current_idx) {
    // Recover from post-eviction stranded cursor. Cursor == size is the natural state after
    // the last child fires in this invocation; it's not a wrap signal.
    if (sections[parent_idx].child_cursor > sections[parent_idx].children.size()) {
        sections[parent_idx].child_cursor = 0;
    }

    const auto& children = sections[parent_idx].children;
    const auto it = std::find_if(children.begin(), children.end(),
                                 [&](const auto& kv) { return kv.first == name; });
    if (it != children.end()) {
        const auto pos = static_cast<uint32_t>(std::distance(children.begin(), it));
        current_idx = it->second;
        sections[parent_idx].child_cursor = pos + 1;
    } else {
        const uint32_t insert_pos = sections[parent_idx].child_cursor;
        sections.emplace_back();
        const auto child_idx = static_cast<uint32_t>(sections.size() - 1);
        sections[child_idx].parent_index = parent_idx;
        sections[parent_idx].children.insert(sections[parent_idx].children.begin() + insert_pos,
                                             {name, child_idx});
        sections[parent_idx].child_cursor = insert_pos + 1;
        current_idx = child_idx;
    }

    sections[current_idx].child_cursor = 0;
}

void Profiler::clear(const uint32_t evict_after_ms) {
    const auto now = chrono_clock::now();
    last_clear_time = now;

    if (evict_after_ms != std::numeric_limits<uint32_t>::max()) {
        const auto threshold = std::chrono::milliseconds{evict_after_ms};
        const auto prune = [&](auto& sections) {
            for (auto& parent : sections) {
                std::erase_if(parent.children, [&](const auto& kv) {
                    return now - sections[kv.second].last_seen > threshold;
                });
            }
        };
        prune(cpu_sections);
        prune(gpu_sections);
    }

    current_gpu_section = 0;
    current_cpu_section = 0;
}

void Profiler::cmd_start(const CommandBufferHandle& cmd,
                         const std::string& name,
                         const vk::PipelineStageFlagBits pipeline_stage) {
    assert(query_pool && "num_gpu_timers is 0?");
    std::vector<uint32_t>& pending = pending_gpu_sections[query_pool];

    const std::size_t parent_idx = current_gpu_section;
    enter_child(gpu_sections, parent_idx, name, current_gpu_section);

    GPUSection& sec = gpu_sections[current_gpu_section];
    assert(sec.timestamp_idx == (uint32_t)-1 &&
           "two sections with the same name or missing collect()?");

    // Bail before mutating section state so a failed capture keeps the prior period's stats.
    if ((pending.size() * SW_QUERY_COUNT) + SW_QUERY_COUNT >= query_pool->get_query_count()) {
        SPDLOG_WARN("profiler query pool exhausted ({} queries); skipping section '{}'",
                    query_pool->get_query_count(), name);
        return;
    }

    const auto now = chrono_clock::now();
    if (sec.last_seen < last_clear_time) {
        sec.sum_duration_ns = 0;
        sec.sq_sum_duration_ns = 0;
        sec.num_captures = 0;
    }
    sec.last_seen = now;

    sec.timestamp_idx = pending.size() * SW_QUERY_COUNT;
    cmd->write_timestamp(query_pool, sec.timestamp_idx, pipeline_stage);
    pending.emplace_back(current_gpu_section);
}

void Profiler::cmd_end(const CommandBufferHandle& cmd,
                       const vk::PipelineStageFlagBits pipeline_stage) {
    assert(query_pool && "num_gpu_timers is 0?");
    assert(current_gpu_section != 0 && "missing cmd_start?");
    GPUSection& section = gpu_sections[current_gpu_section];

    if (section.timestamp_idx != (uint32_t)-1) {
        cmd->write_timestamp(query_pool, section.timestamp_idx + 1, pipeline_stage);
        section.timestamp_idx = (uint32_t)-1;
    }

    current_gpu_section = section.parent_index;
}

void Profiler::collect(const bool wait, const bool keep_query_pool) {
    if (!query_pool) {
        return;
    }

    std::vector<uint32_t>& pending = pending_gpu_sections[query_pool];

    if (pending.empty())
        return;

    assert(current_gpu_section == 0 && "cmd_end missing?");

    const auto results =
        wait ? query_pool->wait_get_query_pool_results_64(0, pending.size() * SW_QUERY_COUNT)
             : query_pool->get_query_pool_results_64(0, pending.size() * SW_QUERY_COUNT);

    for (uint32_t i = 0; i < pending.size(); i++) {
        GPUSection& section = gpu_sections[pending[i]];
        const uint64_t duration_ns =
            (results[SW_QUERY_COUNT * i + 1] - results[SW_QUERY_COUNT * i]) * timestamp_period;
        section.sum_duration_ns += duration_ns;
        section.sq_sum_duration_ns += (duration_ns * duration_ns);
        section.num_captures++;
    }

    query_pool->reset(0, pending.size() * SW_QUERY_COUNT);

    if (keep_query_pool) {
        pending.clear();
    } else {
        pending_gpu_sections.erase(query_pool);
    }
}

void Profiler::start(const std::string& name) {
    const std::size_t parent_idx = current_cpu_section;
    enter_child(cpu_sections, parent_idx, name, current_cpu_section);

    CPUSection& sec = cpu_sections[current_cpu_section];
    const auto now = chrono_clock::now();
    if (sec.last_seen < last_clear_time) {
        sec.sum_duration_ns = 0;
        sec.sq_sum_duration_ns = 0;
        sec.num_captures = 0;
    }
    sec.last_seen = now;
    sec.start = now;
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

std::string format_entry_value(const Profiler::ReportEntry& entry) {
    std::string s = fmt::format("{:.04f} (± {:.04f}) ms", entry.duration, entry.std_deviation);
    if (entry.last_seen_ms_ago) {
        s += fmt::format(" (stale, last {} ago)",
                         format_duration(uint64_t(*entry.last_seen_ms_ago * 1e6)));
    }
    return s;
}

std::string to_string(const std::vector<Profiler::ReportEntry>& entries,
                      const std::size_t indent_depth = 0) {
    std::string result = "";
    for (const auto& entry : entries) {
        result += fmt::format("{}{}: {}\n", std::string(2 * indent_depth, ' '), entry.name,
                              format_entry_value(entry));
        result += to_string(entry.children, indent_depth + 1);
    }
    return result;
}

template <typename SectionType>
std::vector<Profiler::ReportEntry>
make_report(const SectionType& section,
            const std::vector<SectionType>& sections,
            const std::chrono::high_resolution_clock::time_point now,
            const std::chrono::high_resolution_clock::time_point last_clear_time) {
    std::vector<Profiler::ReportEntry> report;
    report.reserve(section.children.size());
    for (const auto& [name, idx] : section.children) {
        const SectionType& sub = sections[idx];

        double avg_ms = 0.0;
        double std_ms = 0.0;
        if (sub.num_captures > 0) {
            const double avg = (double)sub.sum_duration_ns / sub.num_captures;
            const double std_dev =
                std::sqrt((double)sub.sq_sum_duration_ns / sub.num_captures - avg * avg);
            avg_ms = avg / 1e6;
            std_ms = std_dev / 1e6;
        }

        std::optional<double> last_seen_ms_ago;
        if (sub.last_seen < last_clear_time) {
            last_seen_ms_ago =
                std::chrono::duration<double, std::milli>(now - sub.last_seen).count();
        }

        report.emplace_back(name, avg_ms, std_ms, last_seen_ms_ago,
                            make_report(sub, sections, now, last_clear_time));
    }
    return report;
}

std::optional<Profiler::Report>
Profiler::set_collect_get_every(const QueryPoolHandle<vk::QueryType::eTimestamp>& query_pool,
                                const uint32_t report_intervall_millis,
                                const uint32_t evict_after_ms) {
    set_query_pool(query_pool);

    collect(false, true);

    std::optional<Profiler::Report> report;

    if (report_intervall.millis() >= report_intervall_millis) {
        report = get_report();
        if (gpu_sections.size() > 1 && report.value().gpu_report.empty()) {
            // there are sections but we got no results yet
            return std::nullopt;
        }
        clear(evict_after_ms);
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
        const std::string str = fmt::format("{}: {}\n", entry.name, format_entry_value(entry));
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
    const auto now = chrono_clock::now();
    report.cpu_report =
        make_report(cpu_sections[current_cpu_section], cpu_sections, now, last_clear_time);
    report.gpu_report =
        make_report(gpu_sections[current_gpu_section], gpu_sections, now, last_clear_time);
    return report;
}

namespace detail {
std::mutex g_default_profiler_mutex;
std::vector<Profiler*> g_default_profiler_stack;
} // namespace detail

ProfilerHandle get_default_profiler() noexcept {
    const std::lock_guard lock{detail::g_default_profiler_mutex};
    if (detail::g_default_profiler_stack.empty()) {
        return nullptr;
    }
    Profiler* const top = detail::g_default_profiler_stack.back();
    return top != nullptr ? top->shared_from_this() : nullptr;
}

ScopedDefaultProfiler::ScopedDefaultProfiler(ProfilerHandle profiler)
    : profiler(std::move(profiler)) {
    const std::lock_guard lock{detail::g_default_profiler_mutex};
    detail::g_default_profiler_stack.push_back(this->profiler.get());
}

ScopedDefaultProfiler::~ScopedDefaultProfiler() {
    const std::lock_guard lock{detail::g_default_profiler_mutex};
    assert(!detail::g_default_profiler_stack.empty());
    assert(detail::g_default_profiler_stack.back() == profiler.get() &&
           "ScopedDefaultProfiler destroyed out of LIFO order");
    detail::g_default_profiler_stack.pop_back();
}

ScopedDefaultProfiler set_default_profiler(ProfilerHandle profiler) {
    return ScopedDefaultProfiler{std::move(profiler)};
}

} // namespace merian
