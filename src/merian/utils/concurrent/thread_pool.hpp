#pragma once

#include "merian/utils/concurrent/concurrent_queue.hpp"

#include <future>
#include <optional>
#include <thread>
#include <vector>

namespace merian {

class ThreadPool {
  public:
    // concurrency sets the number of threads. queue_size controls the size of the task queue.
    // While the task queue is full, submit blocks.
    ThreadPool(const uint32_t concurrency = std::thread::hardware_concurrency());

    ~ThreadPool();

    ThreadPool(ThreadPool& other) = delete;

    ThreadPool(ThreadPool&& other) {
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

    ThreadPool& operator=(const ThreadPool& src) = delete;

    ThreadPool& operator=(ThreadPool&& src) {
        if (this == &src)
            return *this;
        this->~ThreadPool();
        new (this) ThreadPool(std::move(src));
        return *this;
    }

    uint32_t size();

    template <typename T> std::future<T> submit(const std::function<T()>& function) {
        std::shared_ptr<std::packaged_task<T()>> task =
            std::make_shared<std::packaged_task<T()>>(function);
        std::future<T> future = task->get_future();
        tasks.push([task] { (*task)(); });
        return future;
    }

    template <typename T> std::future<T> submit(const std::function<T()>&& function) {
        std::shared_ptr<std::packaged_task<T()>> task =
            std::make_shared<std::packaged_task<T()>>(function);
        std::future<T> future = task->get_future();
        tasks.push([task] { (*task)(); });
        return future;
    }

  private:
    std::vector<std::thread> threads;
    ConcurrentQueue<std::optional<std::function<void()>>> tasks;
};

} // namespace merian
