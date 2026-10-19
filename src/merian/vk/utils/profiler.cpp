#include "merian/vk/utils/profiler.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/command/command_buffer.hpp"

#include "spdlog/spdlog.h"

#include <cmath>
#include <mutex>
#include <numeric>
#include <vector>

namespace merian {

namespace {

constexpr uint32_t SW_QUERY_COUNT = 2;

constexpr std::size_t index(const Profiler::Period period) {
    return static_cast<std::size_t>(period);
}

double total_ms(const std::vector<ProfilerReport::Entry>& entries) {
    return std::transform_reduce(entries.begin(), entries.end(), 0.0, std::plus{},
                                 [](const auto& entry) { return entry.duration_ms; });
}

double total_std_deviation_ms(const std::vector<ProfilerReport::Entry>& entries) {
    return std::sqrt(std::transform_reduce(
        entries.begin(), entries.end(), 0.0, std::plus{},
        [](const auto& entry) { return entry.std_deviation_ms * entry.std_deviation_ms; }));
}

std::string entries_text(const std::vector<ProfilerReport::Entry>& entries,
                         const std::size_t indent_depth) {
    std::string text;
    for (const auto& entry : entries) {
        text += fmt::format("{}{}\n", std::string(2 * indent_depth, ' '), entry.to_text());
        text += entries_text(entry.children, indent_depth + 1);
    }
    return text;
}

nlohmann::json entries_json(const std::vector<ProfilerReport::Entry>& entries) {
    nlohmann::json json = nlohmann::json::array();
    for (const auto& entry : entries) {
        nlohmann::json& item = json.emplace_back(nlohmann::json{
            {"name", entry.name},
            {"duration_ms", entry.duration_ms},
            {"std_deviation_ms", entry.std_deviation_ms},
            {"children", entries_json(entry.children)},
        });
        if (entry.last_seen_ms_ago) {
            item["last_seen_ms_ago"] = *entry.last_seen_ms_ago;
        }
    }
    return json;
}

} // namespace

std::string ProfilerReport::Entry::to_text() const {
    std::string text =
        fmt::format("{}: {:.04f} (± {:.04f}) ms", name, duration_ms, std_deviation_ms);
    if (last_seen_ms_ago) {
        text += fmt::format(" (stale, last {} ago)",
                            format_duration(static_cast<uint64_t>(*last_seen_ms_ago * 1e6)));
    }
    return text;
}

bool ProfilerReport::empty() const {
    return cpu.empty() && gpu.empty();
}

double ProfilerReport::cpu_total_ms() const {
    return total_ms(cpu);
}

double ProfilerReport::cpu_total_std_deviation_ms() const {
    return total_std_deviation_ms(cpu);
}

double ProfilerReport::gpu_total_ms() const {
    return total_ms(gpu);
}

double ProfilerReport::gpu_total_std_deviation_ms() const {
    return total_std_deviation_ms(gpu);
}

std::string ProfilerReport::to_text() const {
    if (empty()) {
        return "no timestamps captured";
    }
    return fmt::format("CPU:\n{}GPU:\n{}", entries_text(cpu, 1), entries_text(gpu, 1));
}

nlohmann::json ProfilerReport::to_json() const {
    return {{"cpu", entries_json(cpu)}, {"gpu", entries_json(gpu)}};
}

Profiler::Profiler(const ContextHandle& context)
    : timestamp_period(context->get_physical_device()->get_device_limits().timestampPeriod) {
    period_start.fill(chrono_clock::now());
    cpu_sections.assign(1, {});
    gpu_sections.assign(1, {});
    gpu_sections.reserve(1024);
    cpu_sections.reserve(1024);

#ifndef MERIAN_PROFILER_ENABLE
    SPDLOG_DEBUG("MERIAN_PROFILER_ENABLE not defined. MERIAN_PROFILE_* macros are disabled.");
#endif
}

Profiler::~Profiler() = default;

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
    // sections usually repeat in the same order
    const uint32_t cursor = sections[parent_idx].child_cursor;
    const auto it = cursor < children.size() && children[cursor].first == name
                        ? children.begin() + cursor
                        : std::find_if(
                              children.begin(), children.end(),
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

void Profiler::Accumulator::add(const uint64_t duration_ns) {
    sum_duration_ns += duration_ns;
    sq_sum_duration_ns += static_cast<double>(duration_ns) * static_cast<double>(duration_ns);
    num_captures++;
}

template <typename SectionType> void Profiler::reset_stale_periods(SectionType& section) const {
    for (std::size_t period = 0; period < PERIOD_COUNT; period++) {
        if (section.last_seen < period_start[period]) {
            section.periods[period] = {};
        }
    }
}

void Profiler::clear(const Period period, const uint32_t evict_after_ms) {
    const auto now = chrono_clock::now();
    period_start[index(period)] = now;

    if (evict_after_ms != std::numeric_limits<uint32_t>::max()) {
        const auto threshold = std::chrono::milliseconds{evict_after_ms};
        const auto prune = [&](auto& sections) {
            for (auto& parent : sections) {
                std::erase_if(parent.children, [&](const auto& kv) {
                    const auto last_seen = sections[kv.second].last_seen;
                    return now - last_seen > threshold &&
                           last_seen < period_start[index(Period::TOTAL)];
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
    assert(query_pool && "missing set_query_pool?");
    std::vector<PendingSection>& pending = pending_gpu_sections[query_pool];

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
    reset_stale_periods(sec);
    sec.last_seen = now;

    sec.timestamp_idx = pending.size() * SW_QUERY_COUNT;
    cmd->write_timestamp(query_pool, sec.timestamp_idx, pipeline_stage);
    pending.emplace_back(current_gpu_section, now);
}

void Profiler::cmd_end(const CommandBufferHandle& cmd,
                       const vk::PipelineStageFlagBits pipeline_stage) {
    assert(query_pool && "missing set_query_pool?");
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

    std::vector<PendingSection>& pending = pending_gpu_sections[query_pool];

    if (pending.empty())
        return;

    assert(current_gpu_section == 0 && "cmd_end missing?");

    const auto results =
        wait ? query_pool->wait_get_query_pool_results_64(0, pending.size() * SW_QUERY_COUNT)
             : query_pool->get_query_pool_results_64(0, pending.size() * SW_QUERY_COUNT);

    for (uint32_t i = 0; i < pending.size(); i++) {
        GPUSection& section = gpu_sections[pending[i].section];
        const uint64_t duration_ns =
            (results[SW_QUERY_COUNT * i + 1] - results[SW_QUERY_COUNT * i]) * timestamp_period;
        for (std::size_t period = 0; period < PERIOD_COUNT; period++) {
            if (pending[i].recorded >= period_start[period]) {
                section.periods[period].add(duration_ns);
            }
        }
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
    reset_stale_periods(sec);
    sec.last_seen = now;
    sec.start = now;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

void Profiler::end() {
    assert(current_cpu_section != 0 && "missing start?");
    std::atomic_signal_fence(std::memory_order_seq_cst);

    CPUSection& section = cpu_sections[current_cpu_section];
    const auto now = chrono_clock::now();
    const uint64_t duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - section.start).count();
    for (Accumulator& accumulator : section.periods) {
        accumulator.add(duration_ns);
    }

    current_cpu_section = section.parent_index;
}

namespace {

template <typename SectionType>
std::vector<ProfilerReport::Entry>
make_report(const SectionType& section,
            const std::vector<SectionType>& sections,
            const std::size_t period,
            const std::chrono::high_resolution_clock::time_point now,
            const std::chrono::high_resolution_clock::time_point since,
            const std::chrono::milliseconds keep_stale) {
    std::vector<ProfilerReport::Entry> report;
    report.reserve(section.children.size());
    for (const auto& [name, idx] : section.children) {
        const SectionType& sub = sections[idx];
        if (sub.last_seen < since && now - sub.last_seen > keep_stale) {
            continue;
        }

        const auto& accumulator = sub.periods[period];
        double avg_ms = 0.0;
        double std_ms = 0.0;
        if (accumulator.num_captures > 0) {
            const double avg =
                static_cast<double>(accumulator.sum_duration_ns) / accumulator.num_captures;
            const double std_dev =
                std::sqrt(accumulator.sq_sum_duration_ns / accumulator.num_captures - avg * avg);
            avg_ms = avg / 1e6;
            std_ms = std_dev / 1e6;
        }

        std::optional<double> last_seen_ms_ago;
        if (sub.last_seen < since) {
            last_seen_ms_ago =
                std::chrono::duration<double, std::milli>(now - sub.last_seen).count();
        }

        report.emplace_back(name, avg_ms, std_ms, last_seen_ms_ago,
                            make_report(sub, sections, period, now, since, keep_stale));
    }
    return report;
}

} // namespace

std::optional<ProfilerReport>
Profiler::set_collect_get_every(const QueryPoolHandle<vk::QueryType::eTimestamp>& query_pool,
                                const uint32_t report_intervall_millis,
                                const uint32_t evict_after_ms) {
    set_query_pool(query_pool);

    collect(false, true);

    std::optional<ProfilerReport> report;

    if (report_intervall.millis() >= report_intervall_millis) {
        report = get_report(Period::INTERVAL, evict_after_ms);
        if (gpu_sections.size() > 1 && report->gpu.empty()) {
            // there are sections but we got no results yet
            return std::nullopt;
        }
        clear(Period::INTERVAL, evict_after_ms);
        report_intervall.reset();
    }

    return report;
}

ProfilerReport Profiler::get_report(const Period period, const uint32_t keep_stale_ms) const {
    ProfilerReport report;
    const auto now = chrono_clock::now();
    const auto since = period_start[index(period)];
    const std::chrono::milliseconds keep_stale{keep_stale_ms};
    report.cpu = make_report(cpu_sections[current_cpu_section], cpu_sections, index(period), now,
                             since, keep_stale);
    report.gpu = make_report(gpu_sections[current_gpu_section], gpu_sections, index(period), now,
                             since, keep_stale);
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
