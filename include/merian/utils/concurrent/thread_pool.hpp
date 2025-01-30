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

    ThreadPool(ThreadPool&& other) noexcept {
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

    ThreadPool& operator=(ThreadPool&& src) noexcept {
        if (this == &src)
            return *this;
        this->~ThreadPool();
        new (this) ThreadPool(std::move(src));
        return *this;
    }

    // Number of threads in this thread pool.
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

    // returns the number of enqueued tasks. Note that the tasks currently being worked on aren't
    // counted.
    std::size_t queue_size();

    // waits until all to this point submitted tasks are finished.
    void wait_idle();

    // waits until the task queue is empty. Note that threads might still work on their last item.
    // To ensure all threads are idling use wait_idle().
    void wait_empty();

  private:
    std::vector<std::thread> threads;
    ConcurrentQueue<std::optional<std::function<void()>>> tasks;
};

using ThreadPoolHandle = std::shared_ptr<ThreadPool>;

} // namespace merian
