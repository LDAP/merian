#include "merian/utils/concurrent/thread_pool.hpp"

#include <cassert>
#include <latch>

namespace merian {

ThreadPool::ThreadPool(const uint32_t concurrency) {
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
    std::latch latch((std::ptrdiff_t)threads.size());

    for (uint32_t i = 0; i < threads.size(); i++) {
        submit<void>([&latch]() { latch.arrive_and_wait(); });
    }
    latch.wait();
}

void ThreadPool::wait_empty() {
    tasks.wait_empty();
}

} // namespace merian
