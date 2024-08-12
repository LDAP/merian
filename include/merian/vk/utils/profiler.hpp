#pragma once

#include "merian/utils/properties.hpp"
#include "merian/utils/stopwatch.hpp"
#include "merian/vk/utils/query_pool.hpp"

#include <optional>

namespace merian {

class ProfileScope;
class ProfileScopeGPU;

/**
 * @brief      A profiler for CPU and GPU code.
 *
 * Prefer to use the MERIAN_PROFILE_* macros which can be enabled and disabled by
 * defining MERIAN_PROFILER_ENABLE.
 *
 * Remember to export MERIAN_PROFILER_ENABLE if you want to use the profiler.
 *
 *
 * while (True) {
 *     auto& frame_data = ring_fences->next_cycle_wait_and_get();
 *     // now the timestamps from iteration i - RING_SIZE are ready

 *     // collects the results of iteration i - RING_SIZE from the GPU
 *     // and resets the query pool
 *     profiler.set_query_pool(frame_data.query_pool);
 *     profiler.collect();
 *
 *     // use profiler...
 *
 *     queue.submit(... fence);
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
  private:
    using chrono_clock = std::chrono::high_resolution_clock;
    friend ProfileScope;
    friend ProfileScopeGPU;

    struct CPUSection {
        // needed for sorting/printing
        chrono_clock::time_point start;
        chrono_clock::time_point end;

        std::size_t parent_index;
        std::unordered_map<std::string, uint32_t> children;

        uint32_t num_captures{0};
        uint64_t sum_duration_ns{0};
        uint64_t sq_sum_duration_ns{0};
    };

    struct GPUSection {
        // the query index for start. end has index + 1.
        // set to -1 if not in the command buffer
        uint32_t timestamp_idx{(uint32_t)-1};
        // for sorting
        uint64_t start;

        std::size_t parent_index;
        std::unordered_map<std::string, uint32_t> children;

        uint32_t num_captures{0};
        uint64_t sum_duration_ns{0};
        uint64_t sq_sum_duration_ns{0};
    };

  public:
    struct ReportEntry {
        std::string name;
        // in ms
        double duration;
        // in ms
        double std_deviation;
        std::vector<ReportEntry> children;
    };

    struct Report {
        std::vector<ReportEntry> cpu_report;
        std::vector<ReportEntry> gpu_report;

        double cpu_total_std_deviation() const {
            if (cpu_report.empty()) {
                return 0;
            }
            return std::sqrt(std::transform_reduce(
                cpu_report.begin(), cpu_report.end(), 0.0, std::plus{},
                [](const auto& v) { return v.std_deviation * v.std_deviation; }));
        }

        double cpu_total() const {
            if (cpu_report.empty()) {
                return 0;
            }
            return std::transform_reduce(cpu_report.begin(), cpu_report.end(), 0.0, std::plus{},
                                         [](const auto& v) { return v.duration; });
        }

        double gpu_total_std_deviation() const {
            if (gpu_report.empty()) {
                return 0;
            }
            return std::sqrt(std::transform_reduce(
                gpu_report.begin(), gpu_report.end(), 0.0, std::plus{},
                [](const auto& v) { return v.std_deviation * v.std_deviation; }));
        }

        double gpu_total() const {
            if (gpu_report.empty()) {
                return 0;
            }
            return std::transform_reduce(gpu_report.begin(), gpu_report.end(), 0.0, std::plus{},
                                         [](const auto& v) { return v.duration; });
        }

        operator bool() const {
            return !cpu_report.empty() || !gpu_report.empty();
        }
    };

  public:
    Profiler(const ContextHandle& context);

    ~Profiler();

    // Clears the profiler
    void clear();

    void set_query_pool(const QueryPoolHandle<vk::QueryType::eTimestamp>& query_pool);

    // Start a GPU section
    void cmd_start(
        const vk::CommandBuffer& cmd,
        const std::string name,
        const vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands);

    // Stop a GPU section
    void cmd_end(
        const vk::CommandBuffer& cmd,
        const vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands);

    // Collects the results from the GPU.
    void collect(const bool wait = false);

    // Start a CPU section
    void start(const std::string& name);

    // Stop a CPU section
    void end();

    Report get_report();

    // Convenience method that sets the next query pool, collects the results then resets query pool
    // (for GPU profiling).
    //
    // Every report_intervall_millis the method returns a profiling report and clears the profiler
    // when resetting. Meaning, means and std deviation were calculated over the report intervall.
    //
    // Note: The profiler is only reset when the GPU results are actually ready, however, that means
    // that the may be already multiple results for the CPU (noticeable when report_intervall_millis
    // == 0).
    std::optional<Report>
    set_collect_get_every(const QueryPoolHandle<vk::QueryType::eTimestamp>& query_pool,
                          const uint32_t report_intervall_millis = 0);

    // returns the report as string
    static std::string get_report_str(const Profiler::Report& report);

    // outputs the report as config
    static void get_cpu_report_as_config(Properties& config, const Profiler::Report& report);
    static void get_gpu_report_as_config(Properties& config, const Profiler::Report& report);

    // outputs the report as config
    static void get_report_as_config(Properties& config, const Profiler::Report& report);

  private:
    const ContextHandle context;
    const float timestamp_period;

    QueryPoolHandle<vk::QueryType::eTimestamp> query_pool;

    Stopwatch report_intervall;

    uint32_t current_cpu_section = 0;
    uint32_t current_gpu_section = 0;

    // 0 is root node
    std::vector<CPUSection> cpu_sections;
    std::vector<GPUSection> gpu_sections;

    uint32_t clear_index = 0;

    struct PerQueryPoolInfo {
        std::vector<uint32_t> pending_gpu_sections;
        uint32_t clear_index;
    };
    std::unordered_map<QueryPoolHandle<vk::QueryType::eTimestamp>, PerQueryPoolInfo>
        query_pool_infos;
};
using ProfilerHandle = std::shared_ptr<Profiler>;

class ProfileScope {
  public:
    ProfileScope(const ProfilerHandle profiler, const std::string& name) : profiler(profiler) {
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
    ProfilerHandle profiler;
#ifndef NDEBUG
    // Detect overlapping regions
    uint32_t section_index;
#endif
};

class ProfileScopeGPU {
  public:
    // Make sure the command buffers stays valid
    ProfileScopeGPU(const ProfilerHandle profiler,
                    const vk::CommandBuffer& cmd,
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
    ProfilerHandle profiler;
    vk::CommandBuffer cmd;
#ifndef NDEBUG
    // Detect overlapping regions
    uint32_t cpu_section_index;
    uint32_t gpu_section_index;
#endif
};

// clang-format off
#ifdef MERIAN_PROFILER_ENABLE
    // Profiles CPU time of this scope
    #define MERIAN_PROFILE_SCOPE(profiler, name) merian::ProfileScope merian_profile_scope(profiler, name)
    // Profiles CPU and GPU time of this scope
    #define MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, name) merian::ProfileScopeGPU merian_profile_scope(profiler, cmd, name)
#else
    #define MERIAN_PROFILE_SCOPE(profiler, name) (void)profiler
    #define MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, name) (void)profiler
#endif
// clang-format on

} // namespace merian
