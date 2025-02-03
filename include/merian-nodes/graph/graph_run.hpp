#pragma once

#include "merian/utils/chrono.hpp"
#include "merian/utils/concurrent/thread_pool.hpp"
#include "merian/vk/command/caching_command_pool.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/shader/shader_compiler.hpp"
#include "merian/vk/sync/semaphore_binary.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/utils/cpu_queue.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <cstdint>

namespace merian_nodes {

using namespace merian;
using namespace std::literals::chrono_literals;

// Manages data of a single graph run.
class GraphRun {
    template <uint32_t> friend class Graph;

  public:
    GraphRun(const uint32_t iterations_in_flight,
             const ThreadPoolHandle& thread_pool,
             const CPUQueueHandle& cpu_queue,
             const ProfilerHandle& profiler,
             const ResourceAllocatorHandle& allocator,
             const QueueHandle& queue,
             const ShaderCompilerHandle& shader_compiler)
        : iterations_in_flight(iterations_in_flight), thread_pool(thread_pool),
          cpu_queue(cpu_queue), profiler(profiler), allocator(allocator), queue(queue),
          shader_compiler(shader_compiler) {

        semaphores.resize(iterations_in_flight);
        semaphore_value.assign(iterations_in_flight, 1);
        for (uint32_t i = 0; i < iterations_in_flight; i++) {
            semaphores[i] = TimelineSemaphore::create(queue->get_context());
        }
    }

    GraphRun(GraphRun& graph_run) = delete;
    GraphRun(GraphRun&& graph_run) = delete;

    GraphRun& operator=(GraphRun& graph_run) = delete;
    GraphRun& operator=(GraphRun&& graph_run) = delete;

    // Enqueues a wait semaphore for the next submit. Note that during a graph run multiple submits
    // might happen.
    void add_wait_semaphore(const BinarySemaphoreHandle& wait_semaphore,
                            const vk::PipelineStageFlags& wait_stage_flags) noexcept {
        cmd_cache->keep_until_pool_reset(wait_semaphore);
        wait_semaphores.push_back(*wait_semaphore);
        wait_stages.push_back(wait_stage_flags);
        wait_values.push_back(0);
    }

    // Enqueues a signal semaphore for the next submit. Note that during a graph run multiple
    // submits might happen.
    void add_signal_semaphore(const BinarySemaphoreHandle& signal_semaphore) noexcept {
        signal_semaphores.push_back(*signal_semaphore);
        signal_values.push_back(0);
    }

    // Enqueues a wait semaphore for the next submit. Note that during a graph run multiple submits
    // might happen.
    void add_wait_semaphore(const TimelineSemaphoreHandle& wait_semaphore,
                            const vk::PipelineStageFlags& wait_stage_flags,
                            const uint64_t value) noexcept {
        cmd_cache->keep_until_pool_reset(wait_semaphore);
        wait_semaphores.push_back(*wait_semaphore);
        wait_stages.push_back(wait_stage_flags);
        wait_values.push_back(value);
    }

    // Enqueues a signal semaphore for the next submit. Note that during a graph run multiple
    // submits might happen.
    void add_signal_semaphore(const TimelineSemaphoreHandle& signal_semaphore,
                              const uint64_t value) noexcept {
        signal_semaphores.push_back(*signal_semaphore);
        signal_values.push_back(value);
    }

    // Enqueues a callback that is executed after the next submit. Note that during a graph run
    // multiple submits might happen.
    void add_submit_callback(
        const std::function<void(const QueueHandle& queue, GraphRun& run)>& callback) noexcept {
        submit_callbacks.push_back(callback);
    }

    // ------------------------------------------------------------------------------------

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

    // Returns the time difference to the last run in seconds.
    // For the first run of a build the difference to the last run in the previous run is returned.
    const std::chrono::nanoseconds& get_time_delta_duration() const noexcept {
        return time_delta;
    }

    // Returns the time difference to the last run in seconds.
    // For the first run of a build the difference to the last run in the previous run is returned.
    double get_time_delta() const noexcept {
        return to_seconds(time_delta);
    }

    // Return elapsed time since graph initialization
    const std::chrono::nanoseconds& get_elapsed_duration() const noexcept {
        return elapsed;
    }

    // Return elapsed time since graph initialization in seconds.
    double get_elapsed() const noexcept {
        return to_seconds(elapsed);
    }

    // Return elapsed time since the last connect()
    const std::chrono::nanoseconds& get_elapsed_since_connect_duration() const noexcept {
        return elapsed_since_connect;
    }

    // Return elapsed time since graph initialization in seconds.
    double get_elapsed_since_connect() const noexcept {
        return to_seconds(elapsed_since_connect);
    }

    // ------------------------------------------------------------------------------------

    // Returns the profiler that is attached to this run.
    //
    // Can be nullptr if profiling is disabled!
    const ProfilerHandle& get_profiler() const {
        return profiler;
    }

    const ResourceAllocatorHandle& get_allocator() const {
        return allocator;
    }

    const ThreadPoolHandle& get_thread_pool() const {
        return thread_pool;
    }

    const CPUQueueHandle& get_cpu_queue() const {
        return cpu_queue;
    }

    const ShaderCompilerHandle& get_shader_compiler() const {
        return shader_compiler;
    }

    // ------------------------------------------------------------------------------------
    // Interact with graph runtime

    // Hint the graph that waiting was necessary for external events. This information can be used
    // to shift CPU processing back to reduce waiting and reduce latency.
    void hint_external_wait_time(auto chrono_duration) {
        external_wait_time = std::max(external_wait_time, chrono_duration);
    }

    void request_reconnect() noexcept {
        needs_reconnect = true;
    }

    // ------------------------------------------------------------------------------------

    const CommandBufferHandle& get_cmd() {
        assert(cmd && "can only be called in Node::process()");
        return cmd;
    }

    // ------------------------------------------------------------------------------------

    // Queues the callback to be called when the commandbuffer until this point has finished
    // executing on the GPU. Calling this might trigger a GPU submit but graph is free to delay
    // execution of the callback until the end of the run.
    void sync_to_cpu(const std::function<void()>& callback) {
        add_signal_semaphore(semaphores[get_in_flight_index()],
                             semaphore_value[get_in_flight_index()]);
        cpu_queue->submit(semaphores[get_in_flight_index()], semaphore_value[get_in_flight_index()],
                          callback);
        semaphore_value[get_in_flight_index()]++;
    }

    // Queues the callback to be called when the commandbuffer until this point has finished
    // executing on the GPU. Calling this might trigger a GPU submit but graph is free to delay
    // execution of the callback until the end of the run.
    void sync_to_cpu(const std::function<void()>&& callback) {
        add_signal_semaphore(semaphores[get_in_flight_index()],
                             semaphore_value[get_in_flight_index()]);
        cpu_queue->submit(semaphores[get_in_flight_index()], semaphore_value[get_in_flight_index()],
                          callback);
        semaphore_value[get_in_flight_index()]++;
    }

    // Queues the callback to be called when the commandbuffer until this point has finished
    // executing on the GPU. GPU processing will be automatically continued when this callback
    // finishes executing.
    // 
    // Note: This can only be used if there is no present operation depending on the CPU execution.
    void sync_to_cpu_and_back(const std::function<void()>& callback) {
        add_signal_semaphore(semaphores[get_in_flight_index()],
                             semaphore_value[get_in_flight_index()]);
        cmd->end();
        submit();
        cmd = cmd_cache->create_and_begin();
        cpu_queue->submit(semaphores[get_in_flight_index()], semaphore_value[get_in_flight_index()],
                          semaphores[get_in_flight_index()],
                          semaphore_value[get_in_flight_index()] + 1, callback);
        add_wait_semaphore(semaphores[get_in_flight_index()], vk::PipelineStageFlagBits::eTopOfPipe,
                           semaphore_value[get_in_flight_index()] + 1);
        semaphore_value[get_in_flight_index()] += 2;
    }

    // Queues the callback to be called when the commandbuffer until this point has finished
    // executing on the GPU. GPU processing will be automatically continued when this callback
    // finishes executing.
    // 
    // Note: This can only be used if there is no present operation depending on the CPU execution.
    void sync_to_cpu_and_back(const std::function<void()>&& callback) {
        add_signal_semaphore(semaphores[get_in_flight_index()],
                             semaphore_value[get_in_flight_index()]);
        cmd->end();
        submit();
        cmd = cmd_cache->create_and_begin();
        cpu_queue->submit(semaphores[get_in_flight_index()], semaphore_value[get_in_flight_index()],
                          semaphores[get_in_flight_index()],
                          semaphore_value[get_in_flight_index()] + 1, callback);
        add_wait_semaphore(semaphores[get_in_flight_index()], vk::PipelineStageFlagBits::eTopOfPipe,
                           semaphore_value[get_in_flight_index()] + 1);
        semaphore_value[get_in_flight_index()] += 2;
    }

  private:
    void begin_run(const std::shared_ptr<CachingCommandPool>& cmd_cache,
                   const uint64_t iteration,
                   const uint64_t total_iteration,
                   const uint32_t in_flight_index,
                   const std::chrono::nanoseconds& time_delta,
                   const std::chrono::nanoseconds& elapsed,
                   const std::chrono::nanoseconds& elapsed_since_connect) {
        this->cmd_cache = cmd_cache;
        this->iteration = iteration;
        this->total_iteration = total_iteration;
        this->in_flight_index = in_flight_index;
        this->time_delta = time_delta;
        this->elapsed = elapsed;
        this->elapsed_since_connect = elapsed_since_connect;

        external_wait_time = 0ns;

        cmd = cmd_cache->create_and_begin();
    }

    /**
     * @brief      Ends a run by submitting the last commandbuffer to the GPU.
     *
     * @param[in]  fence  The fence to signal when the submitted work completes.
     */
    void end_run(const vk::Fence& fence) {
        cmd->end();
        submit(fence);
        cmd.reset();
    }

    void submit(const vk::Fence& fence = VK_NULL_HANDLE) {
        {
            MERIAN_PROFILE_SCOPE(profiler, "submit");
            queue->submit(get_cmd(), fence, signal_semaphores, wait_semaphores, wait_stages,
                          vk::TimelineSemaphoreSubmitInfo{wait_values, signal_values});
        }

        {
            MERIAN_PROFILE_SCOPE(profiler, "execute submit callbacks");
            for (const auto& callback : submit_callbacks) {
                callback(queue, *this);
            }
        }

        wait_semaphores.clear();
        wait_stages.clear();
        wait_values.clear();
        signal_semaphores.clear();
        signal_values.clear();
        submit_callbacks.clear();
    }

  private:
    const uint32_t iterations_in_flight;
    const ThreadPoolHandle thread_pool;
    const CPUQueueHandle cpu_queue;
    const ProfilerHandle profiler;
    const ResourceAllocatorHandle allocator;
    const QueueHandle queue;
    const ShaderCompilerHandle shader_compiler;

    std::shared_ptr<CachingCommandPool> cmd_cache = nullptr;
    CommandBufferHandle cmd = nullptr;

    std::vector<TimelineSemaphoreHandle> semaphores;
    std::vector<uint64_t> semaphore_value;

    std::vector<vk::Semaphore> wait_semaphores;
    std::vector<uint64_t> wait_values;
    std::vector<vk::PipelineStageFlags> wait_stages;
    std::vector<vk::Semaphore> signal_semaphores;
    std::vector<uint64_t> signal_values;
    std::vector<std::function<void(const QueueHandle& queue, GraphRun& run)>> submit_callbacks;

    std::chrono::nanoseconds external_wait_time;

    bool needs_reconnect = false;
    uint64_t iteration;
    uint64_t total_iteration;
    uint32_t in_flight_index;
    std::chrono::nanoseconds time_delta;
    std::chrono::nanoseconds elapsed;
    std::chrono::nanoseconds elapsed_since_connect;
};

} // namespace merian_nodes
