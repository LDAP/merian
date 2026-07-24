#pragma once

#include "merian/vk/command/caching_command_pool.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/sync/semaphore_binary.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/utils/cpu_queue.hpp"

#include <functional>
#include <memory>

namespace merian {

class Submission;
using SubmissionHandle = std::shared_ptr<Submission>;

// A chain of queue submissions being assembled: commands, wait/signal semaphores and
// submit-time callbacks. Owns its (caching) command pool.
class Submission {
  public:
    Submission(const ContextHandle& context,
               const QueueHandle& queue,
               const CPUQueueHandle& cpu_queue)
        : queue(queue), cpu_queue(cpu_queue),
          pool(std::make_shared<CachingCommandPool>(CommandPool::create(queue))),
          cpu_sync_semaphore(TimelineSemaphore::create(context)) {}

    Submission(const Submission&) = delete;
    Submission& operator=(const Submission&) = delete;

    // Caller guarantees that the GPU finished all work submitted from this Submission.
    void reset() {
        assert(!cmd && "reset() with recording in progress");
        pool->reset();
    }

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

    // Called after every submit of this Submission.
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

    // Queues the callback to be called when the commands recorded until this point have finished
    // executing on the GPU. The execution of the callback may be delayed until after finish().
    void sync_to_cpu(const std::function<void()>& callback) {
        add_signal_semaphore(cpu_sync_semaphore, cpu_sync_value);
        cpu_queue->submit(cpu_sync_semaphore, cpu_sync_value, callback);
        cpu_sync_value++;
    }

    // Queues the callback to be called when the commands recorded until this point have finished
    // executing on the GPU; GPU processing continues when the callback returns. Submits the
    // recording so far.
    //
    // Note: This must not be used if a present operation depends on the CPU execution.
    void sync_to_cpu_and_back(const std::function<void()>& callback) {
        add_signal_semaphore(cpu_sync_semaphore, cpu_sync_value);
        submit();
        cpu_queue->submit(cpu_sync_semaphore, cpu_sync_value, cpu_sync_semaphore,
                          cpu_sync_value + 1, callback);
        add_wait_semaphore(cpu_sync_semaphore, vk::PipelineStageFlagBits::eTopOfPipe,
                           cpu_sync_value + 1);
        cpu_sync_value += 2;
    }

    // ------------------------------------------------------------------------------------

    // Submits the recording so far with the accumulated semaphores; recording then continues on
    // a fresh command buffer.
    void submit(const vk::Fence& fence = VK_NULL_HANDLE) {
        get_cmd()->end();
        queue->submit(cmd, fence, signal_semaphores, wait_semaphores, wait_stages,
                      vk::TimelineSemaphoreSubmitInfo{wait_values, signal_values});
        cmd.reset();

        for (const auto& callback : submit_callbacks) {
            callback(queue, *this);
        }

        wait_semaphores.clear();
        wait_stages.clear();
        wait_values.clear();
        signal_semaphores.clear();
        signal_values.clear();
        submit_callbacks.clear();
    }

    // Runs the pre-submit callbacks and submits the final batch.
    void finish(const vk::Fence& fence = VK_NULL_HANDLE) {
        for (const auto& callback : pre_submit_callbacks) {
            callback(*this);
        }
        pre_submit_callbacks.clear();
        submit(fence);
    }

  private:
    const QueueHandle queue;
    const CPUQueueHandle cpu_queue;
    const std::shared_ptr<CachingCommandPool> pool;

    CommandBufferHandle cmd = nullptr;

    TimelineSemaphoreHandle cpu_sync_semaphore;
    uint64_t cpu_sync_value = 1;

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
