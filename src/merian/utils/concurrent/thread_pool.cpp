#include "merian/utils/concurrent/thread_pool.hpp"

#include <cassert>
#include <spdlog/spdlog.h>

namespace merian {

ThreadPool::ThreadPool(const uint32_t concurrency)
    : wait_idle_barrier((ptrdiff_t)concurrency + 1, []() noexcept {}) {
    assert(concurrency);

    for (uint32_t i = threads.size(); i < concurrency; i++) {
        threads.emplace_back([&] {
            while (true) {
                const std::optional<std::function<void()>> task = tasks.pop();
                if (task) {
                    (*task)();
                } else {
                    return;
                }
            }
        });
    }
}

ThreadPool::ThreadPool(ThreadPool&& other) noexcept
    : wait_idle_barrier((ptrdiff_t)other.threads.size() + 1, []() noexcept {}) {
    tasks = std::move(other.tasks);

    for (uint32_t i = threads.size(); i < other.threads.size(); i++) {
        threads.emplace_back([&] {
            while (true) {
                const std::optional<std::function<void()>> task = tasks.pop();
                if (task) {
                    task.value()();
                } else {
                    return;
                }
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    for (uint32_t i = 0; i < threads.size(); i++) {
        tasks.push(std::nullopt);
    }
    for (auto& t : threads) {
        t.join();
    }
}

uint32_t ThreadPool::size() {
    return threads.size();
}

std::size_t ThreadPool::queue_size() {
    return tasks.size();
}

void ThreadPool::wait_idle() {
    for (uint32_t i = 0; i < threads.size(); i++) {
        submit<void>([this]() { wait_idle_barrier.arrive_and_wait(); });
    }
    wait_idle_barrier.arrive_and_wait();
}

void ThreadPool::wait_empty() {
    tasks.wait_empty();
}

} // namespace merian
