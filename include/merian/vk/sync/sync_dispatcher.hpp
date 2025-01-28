#pragma once

#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/utils/check_result.hpp"

namespace merian {

/* A Dispatches callbacks after dependent on a semaphore value.
 *
 * The callbacks run on the contexts' thread pool.
 */
class SyncDispatcher {

  public:
    SyncDispatcher(const ContextHandle& context)
        : context(context), interrupt_semaphore(TimelineSemaphore::create(context)) {
        dispatcher_thread = std::thread([&]() {
            SPDLOG_DEBUG("dispatcher thread started");

            vk::SemaphoreWaitInfoKHR wait_info{vk::SemaphoreWaitFlagBits::eAny, {}, {}, {}};

            while (!stop || !pending_semaphores.empty()) {

                check_result(context->device.waitSemaphores(wait_info, ~0ul));
            }

            SPDLOG_DEBUG("dispatcher thread quitting");
        });
    }

    ~SyncDispatcher() {
        SPDLOG_DEBUG("stopping dispatcher thread.");
        stop.store(true);
        mtx.lock();
        interrupt_semaphore->signal(pending_values[0]);
        mtx.unlock();

        dispatcher_thread.join();
    }

  private:
    const ContextHandle context;
    const TimelineSemaphoreHandle interrupt_semaphore;

    // always conteins interrupt semaphore at position 0
    std::vector<TimelineSemaphore> pending_semaphores;
    std::vector<vk::Semaphore> pending_vk_semaphores;
    std::vector<uint32_t> pending_values;
    std::mutex mtx;

    std::thread dispatcher_thread;
    std::atomic_bool stop = false;
};

} // namespace merian
