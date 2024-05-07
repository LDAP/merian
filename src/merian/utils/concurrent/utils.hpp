#pragma once

#include "merian/utils/concurrent/thread_pool.hpp"

#include <cstdint>
#include <functional>
#include <thread>

namespace merian {

// Run a function `count` times split into 'tasks' tasks.
// The function gets the index [0,count) and the task index
// [0,concurrency).
inline void parallel_for(const uint32_t count,
                         const std::function<void(uint32_t index, uint32_t thread_index)> function,
                         ThreadPool& thread_pool,
                         const uint32_t tasks = std::thread::hardware_concurrency()) {
    if (count == 0)
        return;

    std::vector<std::future<void>> futures;
    uint32_t real_concurrency = std::min(count, tasks);
    uint32_t count_per_thread = (count + real_concurrency - 1) / real_concurrency;

    for (uint32_t thread_index = 0; thread_index < real_concurrency; thread_index++) {
        futures.emplace_back(
            thread_pool.submit<void>([thread_index, count_per_thread, count, &function]() {
                for (uint32_t index = thread_index * count_per_thread;
                     index < std::min((thread_index + 1) * count_per_thread, count); index++) {
                    function(index, thread_index);
                }
            }));
    }

    for (auto& future : futures) {
        future.get();
    }
}

// Run a function `count` times split into 'tasks' tasks.
// The function gets the index [0,count) and the task index
// [0,concurrency).
inline void parallel_for(const uint32_t count,
                         const std::function<void(uint32_t index, uint32_t thread_index)> function,
                         const uint32_t tasks = std::thread::hardware_concurrency()) {
    ThreadPool thread_pool(std::thread::hardware_concurrency());
    parallel_for(count, function, thread_pool, tasks);
}

} // namespace merian
