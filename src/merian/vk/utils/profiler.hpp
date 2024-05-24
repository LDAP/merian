#pragma once

#include "merian/utils/stopwatch.hpp"
#include "merian/vk/extension/extension.hpp"

#include <optional>
#include <queue>

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
 * Intended use is together with RingFence, since
 * in GPU processing there are often multiple frames-in-flight.
 * Waiting for the frame to be finished to collect the timestamps would flush the pipeline.
 * Use a profiler for every frame-in-flight instead, and collect the results after a few iterations.
 *
 * while (True) {
 *     auto& frame_data = ring_fences->next_cycle_wait_and_get();
 *     // now the timestamps from iteration i - RING_SIZE are ready
 *
 *     //get the profiler for the current iteration
 *     profiler = frame_data.user_data.profiler
 *     // collects the results of iteration i - RING_SIZE from the gpu
 *     // and resets the query pool
 *     profiler.collect();
 *     // print, display results...
 *
 *     profiler.reset(cmd);
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

        operator bool() const {
            return !cpu_report.empty() || !gpu_report.empty();
        }
    };

  public:
    // The timestamps for GPU profiling must be preallocated, therefore you can only capture
    // num_gpu_timers many timers. Throws runtime error if the device or queue does not support
    // timestamp queries.
    //
    // Set num_gpu_timers to 0 to disable GPU profiling capabilities.
    Profiler(const SharedContext context,
             const QueueHandle queue,
             const uint32_t num_gpu_timers = 1024);

  public:
    ~Profiler();

    // Resets the profiler, allowing to capture new timestamps on the GPU.
    // This MUST be called before any cmd_*
    // If clear is true the averages are reset too.
    void cmd_reset(const vk::CommandBuffer& cmd, const bool clear = false);

    // Start a GPU section
    void cmd_start(
        const vk::CommandBuffer& cmd,
        const std::string name,
        const vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eTopOfPipe);

    // Stop a GPU section
    void cmd_end(
        const vk::CommandBuffer& cmd,
        const vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eBottomOfPipe);

    // Collects the results from the GPU.
    void collect(const bool wait = false);

    // Start a CPU section
    void start(const std::string& name);

    // Stop a CPU section
    void end();

    Report get_report();

    // Convenience method that collects the results then resets the profiler (for GPU profiling).
    //
    // Every report_intervall_millis the method returns a profiling report and clears the profiler
    // when resetting. Meaning, means and std deviation were calculated over the report intervall.
    std::optional<Report> collect_reset_get_every(const vk::CommandBuffer& cmd,
                                                  const uint32_t report_intervall_millis = 0);

    // returns the report as string
    static std::string get_report_str(const Profiler::Report& report);

    // renders the report to imgui
    static void get_report_imgui(const Profiler::Report& report);

  private:
    const SharedContext context;
    const uint32_t num_gpu_timers;
    const float timestamp_period;
    vk::QueryPool query_pool{nullptr};
    Stopwatch report_intervall;

    uint32_t current_cpu_section = 0;
    uint32_t current_gpu_section = 0;

    // 0 is root node
    std::vector<CPUSection> cpu_sections;
    std::vector<GPUSection> gpu_sections;

    // sections that have timestamps in the command buffer
    std::vector<uint32_t> pending_gpu_sections;
    bool reset_was_called = false;
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
