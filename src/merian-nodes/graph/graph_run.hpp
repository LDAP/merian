#pragma once

#include "merian/vk/sync/semaphore_binary.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <cstdint>

namespace merian_nodes {

using namespace merian;

// Manages data of a single graph run.
class GraphRun {
    template <uint32_t> friend class Graph;

  public:
    GraphRun(const uint32_t ring_size) : ring_size(ring_size) {}

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

    void add_submit_callback(std::function<void(const QueueHandle& queue)> callback) noexcept {
        submit_callbacks.push_back(callback);
    }

    void request_reconnect() {
        needs_reconnect = true;
    }

    // increases with each run, resets at rebuild
    const uint64_t& get_iteration() const noexcept {
        return iteration;
    }

    // returns the current in-flight index i, with 0 <= i < get_ring_size().
    // It is guaranteed that processing of the last iteration with that index has finished.
    const uint32_t& get_in_flight_index() const noexcept {
        return in_flight_index;
    }

    // returns the number of iterations that might be in flight at a certain time.
    const uint32_t& get_ring_size() const noexcept {
        return ring_size;
    }

    const CommandPoolHandle get_cmd_pool() noexcept {
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
    const std::vector<std::function<void(const QueueHandle& queue)>>
    get_submit_callbacks() const noexcept {
        return submit_callbacks;
    }

    // Call this after you submitted the graph command buffer
    void execute_callbacks(const QueueHandle& queue) const {
        for (auto& callback : submit_callbacks) {
            callback(queue);
        }
    }

    // Returns the profiler that is attached to this run.
    // Can be nullptr if profiling is disabled!
    const ProfilerHandle get_profiler() const {
        return profiler;
    }

  private:
    void reset(const uint64_t iteration,
               const uint32_t in_flight_index,
               const ProfilerHandle profiler,
               const CommandPoolHandle& cmd_pool) {
        this->iteration = iteration;
        this->in_flight_index = in_flight_index;
        this->cmd_pool = cmd_pool;
        wait_semaphores.clear();
        wait_stages.clear();
        wait_values.clear();
        signal_semaphores.clear();
        signal_values.clear();
        submit_callbacks.clear();

        this->profiler = profiler;
        this->needs_reconnect = false;
    }

  private:
    const uint32_t ring_size;

    std::vector<vk::Semaphore> wait_semaphores;
    std::vector<uint64_t> wait_values;
    std::vector<vk::PipelineStageFlags> wait_stages;
    std::vector<vk::Semaphore> signal_semaphores;
    std::vector<uint64_t> signal_values;

    std::vector<std::function<void(const QueueHandle& queue)>> submit_callbacks;

    ProfilerHandle profiler = nullptr;
    CommandPoolHandle cmd_pool = nullptr;
    bool needs_reconnect = false;
    uint64_t iteration;
    uint32_t in_flight_index;
};

} // namespace merian_nodes
