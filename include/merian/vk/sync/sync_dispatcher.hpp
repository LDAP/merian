#pragma once

#include "merian/vk/sync/semaphore_timeline.hpp"

namespace merian {

/* A Dispatches callbacks after dependent on a semaphore value.
 * 
 * The callbacks run on the contexts' thread pool.
 */
class SyncDispatcher {

  public:
    SyncDispatcher(const ContextHandle& context) : context(context), interrupt_semaphore(TimelineSemaphore::create(context)) {
        dispatcher_thread = std::thread([](){
            SPDLOG_DEBUG("dispatcher thread started");

            SPDLOG_DEBUG("dispatcher thread quitting");

        });


    }

    ~SyncDispatcher() {
        SPDLOG_DEBUG("stopping dispatcher thread.");
        stop.store(true);
        interrupt_semaphore->signal(interrupt_wait_value);
        dispatcher_thread.join();
    }

  private:
    const ContextHandle context;
    const TimelineSemaphoreHandle interrupt_semaphore;
    std::thread dispatcher_thread;
    std::atomic_bool stop = false;
    std::atomic<uint64_t> interrupt_wait_value = 1;
};

} // namespace merian
