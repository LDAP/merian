#pragma once

#include "merian/vk/command/caching_command_pool.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/sync/semaphore_binary.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"

#include <deque>
#include <functional>
#include <memory>
#include <optional>

namespace merian {

class Submission;
using SubmissionHandle = std::shared_ptr<Submission>;

// A chain of queue submissions being assembled: commands, wait/signal semaphores, submit-time and
// host callbacks. Owns its (caching) command pool.
class Submission {
  public:
    // IN_ORDER callbacks run in order on the thread that runs the callbacks; ASYNC ones are handed
    // to an AsyncRunner and run concurrently with it.
    enum class CallbackMode : uint8_t { IN_ORDER, ASYNC };
    using AsyncRunner = std::function<void(std::function<void()>&& callback)>;

    Submission(const ContextHandle& context, const QueueHandle& queue)
        : queue(queue), pool(std::make_shared<CachingCommandPool>(CommandPool::create(queue))),
          cpu_sync_semaphore(TimelineSemaphore::create(context)) {}

    Submission(const Submission&) = delete;
    Submission& operator=(const Submission&) = delete;

    // Caller guarantees that the GPU finished all work submitted from this Submission. Runs the
    // remaining host callbacks.
    void reset(const AsyncRunner& run_async = {});

    const QueueHandle& get_queue() const noexcept {
        return queue;
    }

    const CommandBufferHandle& get_cmd() {
        if (!cmd) {
            cmd = pool->create_and_begin();
        }
        return cmd;
    }

    // ------------------------------------------------------------------------------------

    void add_wait_semaphore(const BinarySemaphoreHandle& wait_semaphore,
                            const vk::PipelineStageFlags& wait_stage_flags) noexcept {
        pool->keep_until_pool_reset(wait_semaphore);
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
        pool->keep_until_pool_reset(wait_semaphore);
        wait_semaphores.push_back(*wait_semaphore);
        wait_stages.push_back(wait_stage_flags);
        wait_values.push_back(value);
    }

    void add_signal_semaphore(const TimelineSemaphoreHandle& signal_semaphore,
                              const uint64_t value) noexcept {
        signal_semaphores.push_back(*signal_semaphore);
        signal_values.push_back(value);
    }

    // Wait on / signal a timeline semaphore owned elsewhere. The caller keeps it alive until the
    // submission finished.
    void add_wait_semaphore(const vk::Semaphore wait_semaphore,
                            const vk::PipelineStageFlags& wait_stage_flags,
                            const uint64_t value) noexcept {
        wait_semaphores.push_back(wait_semaphore);
        wait_stages.push_back(wait_stage_flags);
        wait_values.push_back(value);
    }

    void add_signal_semaphore(const vk::Semaphore signal_semaphore, const uint64_t value) noexcept {
        signal_semaphores.push_back(signal_semaphore);
        signal_values.push_back(value);
    }

    // Called after finish() submitted.
    void add_submit_callback(const std::function<void(const QueueHandle& queue,
                                                      Submission& submission)>& callback) noexcept {
        submit_callbacks.push_back(callback);
    }

    // Called once in finish(), after all commands are recorded but before the final submit.
    void
    add_pre_submit_callback(const std::function<void(Submission& submission)>& callback) noexcept {
        pre_submit_callbacks.push_back(callback);
    }

    // ------------------------------------------------------------------------------------

    // Runs callback on the host once the commands recorded so far have finished, from
    // run_finished_callbacks() or reset() on the thread that calls them. A callback must not wait
    // for the GPU.
    void sync_to_cpu(std::function<void()> callback,
                     const CallbackMode mode = CallbackMode::IN_ORDER);

    // As sync_to_cpu; the commands recorded afterwards wait until callback returned. Submits the
    // recording so far. Whoever waits for later work on the queue must run the callbacks meanwhile.
    //
    // Note: This must not be used if a present operation depends on the CPU execution.
    void sync_to_cpu_and_back(std::function<void()> callback);

    // Runs, in order, the callbacks whose commands have finished, without waiting. ASYNC callbacks
    // go to run_async, without it they run in order as well.
    void run_finished_callbacks(const AsyncRunner& run_async = {});

    // Runs the callbacks of the submitted commands, waiting for each.
    void run_submitted_callbacks(const AsyncRunner& run_async = {});

    // The semaphore value the next pending callback waits for, if a submitted one is pending.
    std::optional<std::pair<TimelineSemaphoreHandle, uint64_t>> next_callback_wait() const;

    // Whether submitted commands wait for a callback of sync_to_cpu_and_back that has not run.
    bool waits_for_host() const;

    // ------------------------------------------------------------------------------------

    // Submits the recording so far with the accumulated wait semaphores; recording continues on a
    // fresh command buffer, which the reference get_cmd() returned then holds. The signal
    // semaphores wait for finish().
    void submit();

    // Runs the pre-submit callbacks and submits the final batch with the signal semaphores.
    void finish();

  private:
    void submit_recording(const bool last);

    struct Callback {
        uint64_t value;
        std::function<void()> function;
        CallbackMode mode;
        bool signal_back;
    };

    const QueueHandle queue;
    const std::shared_ptr<CachingCommandPool> pool;

    CommandBufferHandle cmd = nullptr;

    const TimelineSemaphoreHandle cpu_sync_semaphore;
    uint64_t last_cpu_sync_value = 0;
    bool cpu_sync_signal_pending = false;
    uint64_t submitted_cpu_sync_value = 0;
    std::deque<Callback> callbacks;

    std::vector<vk::Semaphore> wait_semaphores;
    std::vector<uint64_t> wait_values;
    std::vector<vk::PipelineStageFlags> wait_stages;
    std::vector<vk::Semaphore> signal_semaphores;
    std::vector<uint64_t> signal_values;
    std::vector<std::function<void(const QueueHandle& queue, Submission& submission)>>
        submit_callbacks;
    std::vector<std::function<void(Submission& submission)>> pre_submit_callbacks;
};

} // namespace merian
