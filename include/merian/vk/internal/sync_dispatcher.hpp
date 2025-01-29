#pragma once

#include "merian/utils/concurrent/thread_pool.hpp"

#include <vulkan/vulkan.hpp>

#include <memory>
#include <queue>

namespace merian {

class Context;
using ContextHandle = std::shared_ptr<Context>;
class TimelineSemaphore;
using TimelineSemaphoreHandle = std::shared_ptr<TimelineSemaphore>;

/* A Dispatches callbacks after dependent on a semaphore value.
 *
 * The callbacks run on the contexts' thread pool.
 */
class CPUDispatcher {

  public:
    CPUDispatcher();

    ~CPUDispatcher();

    void start(ThreadPool& thread_pool, const vk::Device& device);

    void shutdown();

    void submit(const TimelineSemaphoreHandle& semaphore,
                const uint64_t value,
                const std::function<void()>& callback);

    void submit(const TimelineSemaphoreHandle& semaphore,
                const uint64_t value,
                const std::function<void()>&& callback);

  private:
    void wakeup_dispatcher_thread();

  private:
    vk::Device device;
    vk::Semaphore interrupt_semaphore;

    bool running = false;

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

    std::thread dispatcher_thread;
    bool stop = false;
};

} // namespace merian
