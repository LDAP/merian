#pragma once

#include "merian/utils/stopwatch.hpp"
#include "merian/vk/utils/query_pool.hpp"

#include <array>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace merian {

class ProfileScope;
class ProfileScopeGPU;

struct ProfilerReport {
    struct Entry {
        std::string name;
        double duration_ms;
        double std_deviation_ms;
        // set if the section did not run in the period: the time since it last ran
        std::optional<double> last_seen_ms_ago;
        std::vector<Entry> children;

        std::string to_text() const;
    };

    std::vector<Entry> cpu;
    std::vector<Entry> gpu;

    bool empty() const;

    double cpu_total_ms() const;
    double cpu_total_std_deviation_ms() const;
    double gpu_total_ms() const;
    double gpu_total_std_deviation_ms() const;

    std::string to_text() const;

    nlohmann::json to_json() const;
};

/**
 * @brief      A profiler for CPU and GPU code.
 *
 * Prefer to use the MERIAN_PROFILE_* macros which can be enabled and disabled by
 * defining MERIAN_PROFILER_ENABLE.
 *
 * Remember to export MERIAN_PROFILER_ENABLE if you want to use the profiler.
 *
 * With a query pool per iteration in flight:
 *
 * while (true) {
 *     auto& frame_data = ring_fences.next_cycle_wait_get();
 *
 *     // collects the timestamps of the iteration that used the query pool before and resets it
 *     if (const auto report = profiler->set_collect_get_every(frame_data.query_pool, 1000)) {
 *         SPDLOG_INFO("{}", report->to_text());
 *     }
 *
 *     // use profiler...
 *
 *     queue->submit(..., fence);
 * }
 *
 * Does not support overlapping sub-regions. Use two profilers in that case.
 * Example:
 * |--------------|
 * |-------|
 *     |-------|
 *
 */
class Profiler : public std::enable_shared_from_this<Profiler> {
  public:
    // Accumulated independently: INTERVAL restarts with each report, TOTAL only with clear().
    enum class Period : uint8_t {
        INTERVAL,
        TOTAL,
    };

  private:
    static constexpr std::size_t PERIOD_COUNT = 2;

    using chrono_clock = std::chrono::high_resolution_clock;
    friend ProfileScope;
    friend ProfileScopeGPU;

    struct Accumulator {
        uint32_t num_captures{0};
        uint64_t sum_duration_ns{0};
        double sq_sum_duration_ns{0};

        void add(const uint64_t duration_ns);
    };

    struct CPUSection {
        chrono_clock::time_point start;
        chrono_clock::time_point last_seen{};

        std::size_t parent_index;
        // Insertion order tracks execution order; child_cursor is the next insert position.
        std::vector<std::pair<std::string, uint32_t>> children;
        uint32_t child_cursor{0};

        std::array<Accumulator, PERIOD_COUNT> periods;
    };

    struct GPUSection {
        // the query index for start. end has index + 1.
        // set to -1 if not in the command buffer
        uint32_t timestamp_idx{(uint32_t)-1};
        chrono_clock::time_point last_seen{};

        std::size_t parent_index;
        std::vector<std::pair<std::string, uint32_t>> children;
        uint32_t child_cursor{0};

        std::array<Accumulator, PERIOD_COUNT> periods;
    };

  public:
    Profiler(const ContextHandle& context);

    ~Profiler();

    // Restarts the period. Sections not seen within evict_after_ms are dropped unless they hold
    // data of the total period; the default keeps the tree forever.
    void clear(const Period period,
               const uint32_t evict_after_ms = std::numeric_limits<uint32_t>::max());

    void set_query_pool(const QueryPoolHandle<vk::QueryType::eTimestamp>& query_pool);

    // Start a GPU section
    void cmd_start(
        const CommandBufferHandle& cmd,
        const std::string& name,
        const vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands);

    // Stop a GPU section
    void cmd_end(
        const CommandBufferHandle& cmd,
        const vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands);

    // Collects the results from the GPU and resets the query pool.
    void collect(const bool wait = false, const bool keep_query_pool = false);

    // Start a CPU section
    void start(const std::string& name);

    // Stop a CPU section
    void end();

    // Sections that did not run in the period are reported for keep_stale_ms after they last ran.
    ProfilerReport
    get_report(const Period period,
               const uint32_t keep_stale_ms = std::numeric_limits<uint32_t>::max()) const;

    // Convenience method that sets the next query pool, collects the results then resets query pool
    // (for GPU profiling).
    //
    // Every report_intervall_millis the method returns a profiling report and clears the profiler
    // when resetting. Meaning, means and std deviation were calculated over the report intervall.
    //
    // Note: The profiler is only reset when the GPU results are actually ready, however, that means
    // that the may be already multiple results for the CPU (noticeable when report_intervall_millis
    // == 0).
    std::optional<ProfilerReport>
    set_collect_get_every(const QueryPoolHandle<vk::QueryType::eTimestamp>& query_pool,
                          const uint32_t report_intervall_millis = 0,
                          const uint32_t evict_after_ms = std::numeric_limits<uint32_t>::max());

  private:
    template <typename SectionType> void reset_stale_periods(SectionType& section) const;

    // Resolves or inserts `name` as a child of sections[parent_idx]; new children land at
    // the parent's child_cursor so insertion order tracks execution order.
    template <typename SectionType>
    static void enter_child(std::vector<SectionType>& sections,
                            std::size_t parent_idx,
                            const std::string& name,
                            uint32_t& current_idx);

    const float timestamp_period;

    QueryPoolHandle<vk::QueryType::eTimestamp> query_pool;

    Stopwatch report_intervall;

    uint32_t current_cpu_section = 0;
    uint32_t current_gpu_section = 0;

    // 0 is root node
    std::vector<CPUSection> cpu_sections;
    std::vector<GPUSection> gpu_sections;

    // A section is fresh in a period iff last_seen >= the period's start.
    std::array<chrono_clock::time_point, PERIOD_COUNT> period_start;

    struct PendingSection {
        uint32_t section;
        // periods that started later get no result
        chrono_clock::time_point recorded;
    };
    std::unordered_map<QueryPoolHandle<vk::QueryType::eTimestamp>, std::vector<PendingSection>>
        pending_gpu_sections;
};
using ProfilerHandle = std::shared_ptr<Profiler>;

class ProfileScope {
  public:
    ProfileScope(const ProfilerHandle& profiler, const std::string& name) : profiler(profiler) {
        if (!profiler) {
            return;
        }

        profiler->start(name);
#ifndef NDEBUG
        section_index = profiler->current_cpu_section;
#endif
    }

    ~ProfileScope() {
        if (!profiler)
            return;

#ifndef NDEBUG
        assert(section_index == profiler->current_cpu_section && "overlapping profiling sections?");
#endif

        profiler->end();
    }

  private:
    const ProfilerHandle profiler;
#ifndef NDEBUG
    uint32_t section_index = 0;
#endif
};

class ProfileScopeGPU {
  public:
    // Make sure the command buffers stays valid
    ProfileScopeGPU(const ProfilerHandle& profiler,
                    const CommandBufferHandle& cmd,
                    const std::string& name)
        : profiler(profiler), cmd(cmd) {
        if (!profiler)
            return;

        profiler->start(name);
        profiler->cmd_start(cmd, name);

#ifndef NDEBUG
        cpu_section_index = profiler->current_cpu_section;
        gpu_section_index = profiler->current_gpu_section;
#endif
    }

    ~ProfileScopeGPU() {
        if (!profiler)
            return;

#ifndef NDEBUG
        assert(cpu_section_index == profiler->current_cpu_section &&
               "overlapping profiling sections?");
        assert(gpu_section_index == profiler->current_gpu_section &&
               "overlapping profiling sections?");
#endif

        profiler->end();
        profiler->cmd_end(cmd);
    }

  private:
    const ProfilerHandle profiler;
    const CommandBufferHandle& cmd;
#ifndef NDEBUG
    uint32_t cpu_section_index = 0;
    uint32_t gpu_section_index = 0;
#endif
};

ProfilerHandle get_default_profiler() noexcept;

class [[nodiscard]] ScopedDefaultProfiler {
  public:
    explicit ScopedDefaultProfiler(ProfilerHandle profiler);
    ~ScopedDefaultProfiler();

    ScopedDefaultProfiler(const ScopedDefaultProfiler&) = delete;
    ScopedDefaultProfiler& operator=(const ScopedDefaultProfiler&) = delete;
    ScopedDefaultProfiler(ScopedDefaultProfiler&&) = delete;
    ScopedDefaultProfiler& operator=(ScopedDefaultProfiler&&) = delete;

  private:
    ProfilerHandle profiler;
};

[[nodiscard]] ScopedDefaultProfiler set_default_profiler(ProfilerHandle profiler);

// clang-format off
// MERIAN_PP_EXPAND is required for MSVC's traditional preprocessor: without it,
// __VA_ARGS__ is forwarded as a single token group and the overload dispatch
// breaks. Harmless on GCC/Clang.
#define MERIAN_PP_EXPAND(x) x
#define MERIAN_PP_PICK_2(_1, _2, NAME, ...) NAME
#define MERIAN_PP_PICK_3(_1, _2, _3, NAME, ...) NAME

#ifdef MERIAN_PROFILER_ENABLE
    #define MERIAN_PROFILE_SCOPE_1(name) \
        merian::ProfileScope merian_profile_scope(merian::get_default_profiler(), name)
    #define MERIAN_PROFILE_SCOPE_2(profiler, name) \
        merian::ProfileScope merian_profile_scope(profiler, name)
    #define MERIAN_PROFILE_SCOPE(...) \
        MERIAN_PP_EXPAND(MERIAN_PP_PICK_2(__VA_ARGS__, MERIAN_PROFILE_SCOPE_2, MERIAN_PROFILE_SCOPE_1)(__VA_ARGS__))

    #define MERIAN_PROFILE_SCOPE_GPU_2(cmd, name) \
        merian::ProfileScopeGPU merian_profile_scope(merian::get_default_profiler(), cmd, name)
    #define MERIAN_PROFILE_SCOPE_GPU_3(profiler, cmd, name) \
        merian::ProfileScopeGPU merian_profile_scope(profiler, cmd, name)
    #define MERIAN_PROFILE_SCOPE_GPU(...) \
        MERIAN_PP_EXPAND(MERIAN_PP_PICK_3(__VA_ARGS__, MERIAN_PROFILE_SCOPE_GPU_3, MERIAN_PROFILE_SCOPE_GPU_2)(__VA_ARGS__))
#else
    #define MERIAN_PROFILE_DISCARD_1(a)       ((void)sizeof(a))
    #define MERIAN_PROFILE_DISCARD_2(a, b)    ((void)sizeof(a), (void)sizeof(b))
    #define MERIAN_PROFILE_DISCARD_3(a, b, c) ((void)sizeof(a), (void)sizeof(b), (void)sizeof(c))
    #define MERIAN_PROFILE_SCOPE(...) \
        MERIAN_PP_EXPAND(MERIAN_PP_PICK_2(__VA_ARGS__, MERIAN_PROFILE_DISCARD_2, MERIAN_PROFILE_DISCARD_1)(__VA_ARGS__))
    #define MERIAN_PROFILE_SCOPE_GPU(...) \
        MERIAN_PP_EXPAND(MERIAN_PP_PICK_3(__VA_ARGS__, MERIAN_PROFILE_DISCARD_3, MERIAN_PROFILE_DISCARD_2)(__VA_ARGS__))
#endif

// Detailed scopes are intended for fine-grained, potentially per-item profiling
// (e.g. one section per mesh / per BLAS build). They are off by default to keep
// the GPU query pool from being exhausted; enable by defining MERIAN_PROFILER_DETAILED
// (which also requires MERIAN_PROFILER_ENABLE).
#if defined(MERIAN_PROFILER_ENABLE) && defined(MERIAN_PROFILER_DETAILED)
    #define MERIAN_PROFILE_SCOPE_DETAILED(...)     MERIAN_PROFILE_SCOPE(__VA_ARGS__)
    #define MERIAN_PROFILE_SCOPE_GPU_DETAILED(...) MERIAN_PROFILE_SCOPE_GPU(__VA_ARGS__)
#else
    #define MERIAN_PROFILE_DETAILED_DISCARD_1(a)       ((void)sizeof(a))
    #define MERIAN_PROFILE_DETAILED_DISCARD_2(a, b)    ((void)sizeof(a), (void)sizeof(b))
    #define MERIAN_PROFILE_DETAILED_DISCARD_3(a, b, c) ((void)sizeof(a), (void)sizeof(b), (void)sizeof(c))
    #define MERIAN_PROFILE_SCOPE_DETAILED(...) \
        MERIAN_PP_EXPAND(MERIAN_PP_PICK_2(__VA_ARGS__, MERIAN_PROFILE_DETAILED_DISCARD_2, MERIAN_PROFILE_DETAILED_DISCARD_1)(__VA_ARGS__))
    #define MERIAN_PROFILE_SCOPE_GPU_DETAILED(...) \
        MERIAN_PP_EXPAND(MERIAN_PP_PICK_3(__VA_ARGS__, MERIAN_PROFILE_DETAILED_DISCARD_3, MERIAN_PROFILE_DETAILED_DISCARD_2)(__VA_ARGS__))
#endif
// clang-format on

} // namespace merian
