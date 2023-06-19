#pragma once

#include "merian/vk/extension/extension.hpp"

#include <optional>

namespace merian {

class Profiler;
using ProfilerHandle = std::shared_ptr<Profiler>;

class Profiler {
  private:
    using chrono_clock = std::chrono::high_resolution_clock;

    struct CPUSection {
        chrono_clock::time_point start;
        chrono_clock::time_point end;
        uint64_t duration_ns;
        std::string name;
    };

    struct GPUSection {
        uint64_t start{};
        uint64_t end{};
        uint64_t duration_ns;
        std::string name;
        uint32_t start_timestamp_idx;
        uint32_t end_timestamp_idx;
    };

  public:
    static ProfilerHandle make(const SharedContext context, const uint32_t num_gpu_timers = 1028) {
        return std::shared_ptr<Profiler>(new Profiler(context, num_gpu_timers));
    }

  private:
    // The timestamps for GPU profiling must be preallocated, therefore you can only caputure
    // num_gpu_timers many timers.
    Profiler(const SharedContext context, const uint32_t num_gpu_timers = 1028);

  public:
    ~Profiler();

    // Resets the profiler, allowing to capture a new profile
    // this MUST be called before any cmd_*
    void cmd_reset(const vk::CommandBuffer& cmd);

    // Start a GPU section
    uint32_t cmd_start(
        const vk::CommandBuffer& cmd,
        const std::string name,
        const vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands);

    // Stop a GPU section
    void cmd_end(
        const vk::CommandBuffer& cmd,
        const uint32_t start_id,
        const vk::PipelineStageFlagBits pipeline_stage = vk::PipelineStageFlagBits::eAllCommands);

    // Collects the results from the GPU.
    void collect(const bool wait = false);

    // Start a CPU section
    uint32_t start(const std::string& name);

    // Stop a CPU section
    void end(const uint32_t start_id);

    std::string get_report();

  private:
    const SharedContext context;
    uint32_t num_gpu_timers;
    vk::QueryPool query_pool;
    float timestamp_period;

    std::vector<CPUSection> cpu_sections;

    std::vector<GPUSection> gpu_sections;
    // gpu section index, is_end
    std::vector<std::tuple<std::size_t, bool>> pending_gpu_timestamps;
    bool reset_was_called = false;
};

class ProfileScope {
  public:
    ProfileScope(const ProfilerHandle profiler, const std::string& name) : profiler(profiler) {
        if (profiler)
            section_index = profiler->start(name);
    }

    ~ProfileScope() {
        if (profiler)
            profiler->end(section_index);
    }

  private:
    ProfilerHandle profiler;
    uint32_t section_index;
};

class ProfileScopeGPU {
  public:
    // Make sure the command buffers stays valid
    ProfileScopeGPU(const ProfilerHandle profiler, const vk::CommandBuffer& cmd, const std::string& name) : profiler(profiler), cmd(cmd) {
        if (!profiler)
            return;

        cpu_section_index = profiler->start(name);
        gpu_section_index = profiler->cmd_start(cmd, name);
    }

    ~ProfileScopeGPU() {
        if (!profiler)
            return;

        profiler->end(cpu_section_index);
        profiler->cmd_end(cmd, gpu_section_index);
    }

  private:
    ProfilerHandle profiler;
    vk::CommandBuffer cmd;

    uint32_t cpu_section_index;
    uint32_t gpu_section_index;
};

// Profiles CPU time of this scope
#define MERIAN_PROFILE_SCOPE(profiler, name) merian::ProfileScope merian_profile_scope(profiler, name)

// Profiles CPU and GPU time of this scope
#define CMD_MERIAN_PROFILE_SCOPE(profiler, cmd, name) merian::ProfileScopeGPU merian_profile_scope(profiler, cmd, name)

} // namespace merian
