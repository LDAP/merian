#include "merian/vk/internal/sync_dispatcher.hpp"

#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <spdlog/spdlog.h>

namespace merian {

CPUDispatcher::CPUDispatcher() {
    waiting_vk_semaphores.emplace_back();
    waiting_semaphores.emplace_back();
    waiting_values.emplace_back();
    waiting_callbacks.emplace_back([]() { /* dummy */ });
}

void CPUDispatcher::start(ThreadPool& thread_pool, const vk::Device& device) {
    assert(!running);
    assert(waiting_vk_semaphores.size() == 1);

    this->device = device;
    const vk::SemaphoreTypeCreateInfo type_create_info{vk::SemaphoreType::eTimeline, 0};
    interrupt_semaphore = waiting_vk_semaphores[0] =
        device.createSemaphore(vk::SemaphoreCreateInfo{{}, &type_create_info});

    dispatcher_thread = std::thread([&]() {
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
            assert(waiting_vk_semaphores[0] == interrupt_semaphore);

            waiting_values[0] = interrupt_value;

            if (stop && waiting_semaphores.size() == 1) {
                // only the interrupt semaphore is left
                SPDLOG_DEBUG("dispatcher thread quitting");
                mtx.unlock();
                return;
            }

            mtx.unlock();

            wait_info_any.setSemaphores(waiting_vk_semaphores);
            wait_info_any.setValues(waiting_values);

            SPDLOG_TRACE("dispatcher thread waiting for {} semaphores", waiting_semaphores.size());
            check_result(device.waitSemaphores(wait_info_any, ~0ul),
                         "failed waiting for semaphores in SyncDispatcher");
            SPDLOG_TRACE("dispatcher thread woke up");

            for (uint32_t i = 1; i < waiting_semaphores.size();) {
                wait_info_all.setSemaphores(waiting_vk_semaphores[i]);
                wait_info_all.setValues(waiting_values[i]);

                vk::Result result = device.waitSemaphores(wait_info_all, 0);
                if (result == vk::Result::eSuccess) {
                    std::swap(waiting_vk_semaphores[i], waiting_vk_semaphores.back());
                    std::swap(waiting_semaphores[i], waiting_semaphores.back());
                    std::swap(waiting_values[i], waiting_values.back());
                    std::swap(waiting_callbacks[i], waiting_callbacks.back());

                    SPDLOG_TRACE("dispatcher thread submitting to thread pool");
                    thread_pool.submit(std::move(waiting_callbacks.back()));

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

    running = true;
}

void CPUDispatcher::shutdown() {
    assert(running);
    assert(waiting_vk_semaphores.size() == 1);

    SPDLOG_DEBUG("stopping dispatcher thread.");

    mtx.lock();
    stop = true;
    mtx.unlock();

    wakeup_dispatcher_thread();

    dispatcher_thread.join();

    assert(waiting_vk_semaphores.size() == 1);
    device.destroySemaphore(waiting_vk_semaphores[0]);

    running = false;
    stop = false;
}

CPUDispatcher::~CPUDispatcher() {
    assert(!running);
}

void CPUDispatcher::submit(const TimelineSemaphoreHandle& semaphore,
                           const uint64_t value,
                           const std::function<void()>& callback) {
    assert(running);

    mtx.lock();
    pending.emplace(semaphore, value, callback);
    mtx.unlock();

    wakeup_dispatcher_thread();
}

void CPUDispatcher::submit(const TimelineSemaphoreHandle& semaphore,
                           const uint64_t value,
                           const std::function<void()>&& callback) {
    assert(running);

    mtx.lock();
    pending.emplace(semaphore, value, callback);
    mtx.unlock();

    wakeup_dispatcher_thread();
}

void CPUDispatcher::wakeup_dispatcher_thread() {
    mtx.lock();
    device.signalSemaphore(vk::SemaphoreSignalInfo{interrupt_semaphore, interrupt_value++});
    mtx.unlock();
}

} // namespace merian
