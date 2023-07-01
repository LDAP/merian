#pragma once

#include <cstdint>
#include <functional>
#include <thread>

namespace merian {

// Run a function `count` times. The function get the index [0,count) and the thread index
// [0,concurrency).
inline void parallel_for(const uint32_t count,
                         const std::function<void(uint32_t index, uint32_t thread_index)> function,
                         const uint32_t concurrency = std::thread::hardware_concurrency()) {
    if (count == 0)
        return;

    std::vector<std::thread> threads;
    uint32_t real_concurrency = std::min(count, concurrency);
    uint32_t count_per_thread = (count + real_concurrency - 1) / real_concurrency;

    for (uint32_t thread_index = 0; thread_index < real_concurrency; thread_index++) {
        threads.emplace_back([thread_index, count_per_thread, count, &function]() {
            for (uint32_t index = thread_index * count_per_thread;
                 index < std::min((thread_index + 1) * count_per_thread, count); index++) {
                function(index, thread_index);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

} // namespace merian
