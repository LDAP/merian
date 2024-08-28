#pragma once

#include "merian/utils/chrono.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/sync/semaphore_binary.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <cstdint>

namespace merian_nodes {

using namespace merian;
using namespace std::literals::chrono_literals;

// Manages data of a single graph run.
class GraphRun {
    template <uint32_t> friend class Graph;

  public:
    GraphRun(const uint32_t iterations_in_flight) : iterations_in_flight(iterations_in_flight) {}

    void add_wait_semaphore(const BinarySemaphoreHandle& wait_semaphore,
                            const vk::PipelineStageFlags& wait_stage_flags) noexcept {
        wait_semaphores.push_back(*wait_semaphore);
        wait_stages.push_back(wait_stage_flags);
        wait_values.push_back(0);
    }

    void add_signal_semaphore(const BinarySemaphoreHandle& signal_semaphore) noexcept {
        signal_semaphores.push_back(*signal_semaphore);
        signal_values.push_back(0);
    }

    void add_wait_semaphore(const TimelineSemaphoreHandle& wait_semaphore,
                            const vk::PipelineStageFlags& wait_stage_flags,
                            const uint64_t value) noexcept {
        wait_semaphores.push_back(*wait_semaphore);
        wait_stages.push_back(wait_stage_flags);
        wait_values.push_back(value);
    }

    void add_signal_semaphore(const TimelineSemaphoreHandle& signal_semaphore,
                              const uint64_t value) noexcept {
        signal_semaphores.push_back(*signal_semaphore);
        signal_values.push_back(value);
    }

    void add_submit_callback(
        const std::function<void(const QueueHandle& queue, GraphRun& run)>& callback) noexcept {
        submit_callbacks.push_back(callback);
    }

    void request_reconnect() noexcept {
        needs_reconnect = true;
    }

    // Number of iterations since connect.
    // Use get_total_iteration() for iterations since graph initialization.
    //
    // Iterations are 0-indexed.
    const uint64_t& get_iteration() const noexcept {
        return iteration;
    }

    // Number of iterations since graph initialization.
    // Use get_iteration() for iterations since connect.
    //
    // Iterations are 0-indexed.
    const uint64_t& get_total_iteration() const noexcept {
        return total_iteration;
    }

    // returns the current in-flight index i, with 0 <= i < get_iterations_in_flight().
    // It is guaranteed that processing of the last iteration with that index has finished.
    const uint32_t& get_in_flight_index() const noexcept {
        return in_flight_index;
    }

    // returns the number of iterations that might be in flight at a certain time.
    const uint32_t& get_iterations_in_flight() const noexcept {
        return iterations_in_flight;
    }

    const CommandPoolHandle& get_cmd_pool() noexcept {
        return cmd_pool;
    }

    // Add this to the submit call for the graph command buffer
    const std::vector<vk::Semaphore>& get_wait_semaphores() const noexcept {
        return wait_semaphores;
    }

    // Add this to the submit call for the graph command buffer
    const std::vector<vk::PipelineStageFlags>& get_wait_stages() const noexcept {
        return wait_stages;
    }

    // Add this to the submit call for the graph command buffer
    const std::vector<vk::Semaphore>& get_signal_semaphores() const noexcept {
        return signal_semaphores;
    }

    // Add this to the submit call for the graph command buffer
    // The retuned pointer is valid until the next call to run.
    vk::TimelineSemaphoreSubmitInfo get_timeline_semaphore_submit_info() const noexcept {
        return vk::TimelineSemaphoreSubmitInfo{wait_values, signal_values};
    }

    // You must call every callback after you submited the graph command buffer
    // Or you use the execute_callbacks function.
    const std::vector<std::function<void(const QueueHandle& queue, GraphRun& run)>>&
    get_submit_callbacks() const noexcept {
        return submit_callbacks;
    }

    // Call this after you submitted the graph command buffer
    void execute_callbacks(const QueueHandle& queue) {
        for (const auto& callback : submit_callbacks) {
            callback(queue, *this);
        }
    }

    // Returns the profiler that is attached to this run.
    //
    // Can be nullptr if profiling is disabled!
    const ProfilerHandle& get_profiler() const {
        return profiler;
    }

    const ResourceAllocatorHandle& get_allocator() const {
        return allocator;
    }

    // Returns the time difference to the last run in seconds.
    // For the first run of a build the difference to the last run in the previous run is returned.
    const std::chrono::nanoseconds& get_time_delta_duration() const {
        return time_delta;
    }

    // Returns the time difference to the last run in seconds.
    // For the first run of a build the difference to the last run in the previous run is returned.
    double get_time_delta() const {
        return to_seconds(time_delta);
    }

    // Return elapsed time since graph initialization
    const std::chrono::nanoseconds& get_elapsed_duration() const {
        return elapsed;
    }

    // Return elapsed time since graph initialization in seconds.
    double get_elapsed() const {
        return to_seconds(elapsed);
    }

    // Return elapsed time since the last connect()
    const std::chrono::nanoseconds& get_elapsed_since_connect_duration() const {
        return elapsed_since_connect;
    }

    // Return elapsed time since graph initialization in seconds.
    double get_elapsed_since_connect() const {
        return to_seconds(elapsed_since_connect);
    }

    // Hint the graph that waiting was necessary for external events. This information can be used
    // to shift CPU processing back to reduce waiting and reduce latency.
    void hint_external_wait_time(auto chrono_duration) {
        external_wait_time = std::max(external_wait_time, chrono_duration);
    }

  private:
    void reset(const uint64_t iteration,
               const uint32_t in_flight_index,
               const ProfilerHandle& profiler,
               const CommandPoolHandle& cmd_pool,
               const ResourceAllocatorHandle& allocator,
               const std::chrono::nanoseconds time_delta,
               const std::chrono::nanoseconds elapsed,
               const std::chrono::nanoseconds elapsed_run,
               const uint64_t total_iterations) {
        this->iteration = iteration;
        this->in_flight_index = in_flight_index;
        this->cmd_pool = cmd_pool;
        this->allocator = allocator;
        this->time_delta = time_delta;
        this->elapsed = elapsed;
        this->elapsed_since_connect = elapsed_run;
        this->total_iteration = total_iterations;
        wait_semaphores.clear();
        wait_stages.clear();
        wait_values.clear();
        signal_semaphores.clear();
        signal_values.clear();
        submit_callbacks.clear();
        external_wait_time = 0ns;

        this->profiler = profiler;
        this->needs_reconnect = false;
    }

  private:
    const uint32_t iterations_in_flight;

    std::vector<vk::Semaphore> wait_semaphores;
    std::vector<uint64_t> wait_values;
    std::vector<vk::PipelineStageFlags> wait_stages;
    std::vector<vk::Semaphore> signal_semaphores;
    std::vector<uint64_t> signal_values;

    std::vector<std::function<void(const QueueHandle& queue, GraphRun& run)>> submit_callbacks;
    std::chrono::nanoseconds external_wait_time;

    ProfilerHandle profiler = nullptr;
    CommandPoolHandle cmd_pool = nullptr;
    ResourceAllocatorHandle allocator = nullptr;

    bool needs_reconnect = false;
    uint64_t iteration;
    uint64_t total_iteration;
    uint32_t in_flight_index;
    std::chrono::nanoseconds time_delta;
    std::chrono::nanoseconds elapsed;
    std::chrono::nanoseconds elapsed_since_connect;
};

} // namespace merian_nodes
