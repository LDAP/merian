#include "merian/utils/concurrent/thread_pool.hpp"

#include <cassert>

namespace merian {

ThreadPool::ThreadPool(const uint32_t concurrency) {
    assert(concurrency);

    for (uint32_t i = threads.size(); i < concurrency; i++) {
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

} // namespace merian
