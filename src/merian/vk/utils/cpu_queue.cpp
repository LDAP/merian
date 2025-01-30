#include "merian/vk/utils/cpu_queue.hpp"

#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <spdlog/spdlog.h>

namespace merian {

CPUQueue::CPUQueue(const ContextHandle& context, const ThreadPoolHandle& thread_pool)
    : context(context), thread_pool(thread_pool),
      interrupt_semaphore(TimelineSemaphore::create(context)) {

    waiting_vk_semaphores.emplace_back(**interrupt_semaphore);
    waiting_semaphores.emplace_back();
    waiting_values.emplace_back();
    waiting_callbacks.emplace_back([]() { /* dummy */ });

    dispatcher_thread = std::thread([this]() {
        SPDLOG_DEBUG("dispatcher thread started");

        vk::SemaphoreWaitInfoKHR wait_info_any{vk::SemaphoreWaitFlagBits::eAny, {}, {}, {}};
        vk::SemaphoreWaitInfoKHR wait_info_all{{}, {}, {}, {}};

        while (true) {
            mtx.lock();
            while (!pending.empty()) {
                auto& front = pending.front();
                waiting_vk_semaphores.emplace_back(*front.semaphore);
                waiting_semaphores.emplace_back(std::move(front.semaphore));
                waiting_values.emplace_back(front.value);
                waiting_callbacks.emplace_back(std::move(front.callback));

                pending.pop();
            }

            assert(waiting_semaphores[0] == nullptr);
            assert(waiting_vk_semaphores[0] == **interrupt_semaphore);

            waiting_values[0] = interrupt_value;

            if (stop && waiting_semaphores.size() == 1) {
                // only the interrupt semaphore is left
                SPDLOG_DEBUG("dispatcher thread quitting");
                mtx.unlock();
                return;
            }

            if (signal_wait_idle && waiting_semaphores.size() == 1) {
                mtx.unlock();
                this->thread_pool->wait_idle();
                mtx.lock();

                if (pending.empty()) {
                    // check again since tasks of the thread pool can submit new tasks.
                    signal_wait_idle = false;
                    wait_idle_barrier.arrive_and_wait();
                }
            }
            mtx.unlock();

            wait_info_any.setSemaphores(waiting_vk_semaphores);
            wait_info_any.setValues(waiting_values);

            SPDLOG_TRACE("dispatcher thread waiting for {} semaphores", waiting_semaphores.size());
            check_result(this->context->device.waitSemaphores(wait_info_any, ~0ul),
                         "failed waiting for semaphores in SyncDispatcher");
            SPDLOG_TRACE("dispatcher thread woke up");

            for (uint32_t i = 1; i < waiting_semaphores.size();) {
                wait_info_all.setSemaphores(waiting_vk_semaphores[i]);
                wait_info_all.setValues(waiting_values[i]);

                vk::Result result = this->context->device.waitSemaphores(wait_info_all, 0);
                if (result == vk::Result::eSuccess) {
                    std::swap(waiting_vk_semaphores[i], waiting_vk_semaphores.back());
                    std::swap(waiting_semaphores[i], waiting_semaphores.back());
                    std::swap(waiting_values[i], waiting_values.back());
                    std::swap(waiting_callbacks[i], waiting_callbacks.back());

                    SPDLOG_TRACE("dispatcher thread submitting to thread pool");
                    this->thread_pool->submit(std::move(waiting_callbacks.back()));

                    waiting_vk_semaphores.pop_back();
                    waiting_semaphores.pop_back();
                    waiting_values.pop_back();
                    waiting_callbacks.pop_back();
                } else {
                    i++;
                }
            }
        }
    });
}

CPUQueue::~CPUQueue() {
    SPDLOG_DEBUG("stopping dispatcher thread.");

    mtx.lock();
    stop = true;
    interrupt_semaphore->signal(interrupt_value++);
    mtx.unlock();

    dispatcher_thread.join();

    assert(waiting_vk_semaphores.size() == 1);
}

void CPUQueue::submit(const TimelineSemaphoreHandle& semaphore,
                      const uint64_t value,
                      const std::function<void()>& callback) {
    mtx.lock();
    pending.emplace(semaphore, value, callback);
    interrupt_semaphore->signal(interrupt_value++);
    mtx.unlock();
}

void CPUQueue::submit(const TimelineSemaphoreHandle& semaphore,
                      const uint64_t value,
                      const std::function<void()>&& callback) {
    mtx.lock();
    pending.emplace(semaphore, value, callback);
    interrupt_semaphore->signal(interrupt_value++);
    mtx.unlock();
}

void CPUQueue::submit(const TimelineSemaphoreHandle& wait_semaphore,
                      const uint64_t wait_value,
                      const TimelineSemaphoreHandle& signal_semaphore,
                      const uint64_t signal_value,
                      const std::function<void()>& callback) {
    submit(wait_semaphore, wait_value, [signal_semaphore, signal_value, callback]() {
        callback();
        signal_semaphore->signal(signal_value);
    });
}

void CPUQueue::submit(const TimelineSemaphoreHandle& wait_semaphore,
                      const uint64_t wait_value,
                      const TimelineSemaphoreHandle& signal_semaphore,
                      const uint64_t signal_value,
                      const std::function<void()>&& callback) {
    submit(wait_semaphore, wait_value, [signal_semaphore, signal_value, callback]() {
        callback();
        signal_semaphore->signal(signal_value);
    });
}

void CPUQueue::wait_idle() {
    std::lock_guard<std::mutex> lock(wait_idle_mtx);

    mtx.lock();
    signal_wait_idle = true;
    interrupt_semaphore->signal(interrupt_value++);
    mtx.unlock();

    wait_idle_barrier.arrive_and_wait();
}

} // namespace merian
