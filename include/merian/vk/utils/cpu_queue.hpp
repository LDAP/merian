#pragma once

#include "merian/utils/concurrent/thread_pool.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"

#include <barrier>
#include <queue>

namespace merian {

/* Submit work to a CPU thread pool and schedule with GPU work using timeline semaphores.
 *
 * Imitates the Vulkan queue for CPU work: Instead of command buffer a lambda is supplied.
 */
class CPUQueue {

  public:
    CPUQueue(const ContextHandle& context, const ThreadPoolHandle& thread_pool);

    ~CPUQueue();

    void submit(const TimelineSemaphoreHandle& wait_semaphore,
                const uint64_t wait_value,
                const std::function<void()>& callback);

    void submit(const TimelineSemaphoreHandle& wait_semaphore,
                const uint64_t wait_value,
                const std::function<void()>&& callback);

    void submit(const TimelineSemaphoreHandle& wait_semaphore,
                const uint64_t wait_value,
                const TimelineSemaphoreHandle& signal_semaphore,
                const uint64_t signal_value,
                const std::function<void()>& callback);

    void submit(const TimelineSemaphoreHandle& wait_semaphore,
                const uint64_t wait_value,
                const TimelineSemaphoreHandle& signal_semaphore,
                const uint64_t signal_value,
                const std::function<void()>&& callback);

    void wait_idle();

  private:
    const ContextHandle context;
    const ThreadPoolHandle thread_pool;
    const TimelineSemaphoreHandle interrupt_semaphore;

    struct PendingItem {
        TimelineSemaphoreHandle semaphore;
        uint64_t value;
        std::function<void()> callback;
    };

    // written by main thread, read by dispatcher thread
    std::queue<PendingItem> pending;
    uint64_t interrupt_value = 1;

    // always conteins interrupt semaphore at position 0
    std::vector<TimelineSemaphoreHandle> waiting_semaphores;
    std::vector<vk::Semaphore> waiting_vk_semaphores;
    std::vector<uint64_t> waiting_values;
    std::vector<std::function<void()>> waiting_callbacks;

    std::mutex mtx;
    std::mutex wait_idle_mtx;

    std::thread dispatcher_thread;
    bool stop = false;

    bool signal_wait_idle = false;
    std::barrier<void (*)(void) noexcept> wait_idle_barrier{2, []() noexcept {}};
};

using CPUQueueHandle = std::shared_ptr<CPUQueue>;

} // namespace merian
