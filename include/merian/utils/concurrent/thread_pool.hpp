#pragma once

#include "merian/utils/concurrent/concurrent_queue.hpp"

#include <barrier>
#include <future>
#include <optional>
#include <thread>
#include <vector>

namespace merian {

class ThreadPool {
  public:
    // submit blocks while queue_capacity tasks wait.
    ThreadPool(const uint32_t concurrency = std::thread::hardware_concurrency(),
               const uint64_t queue_capacity = UINT64_MAX);

    ~ThreadPool();

    ThreadPool(ThreadPool& other) = delete;

    ThreadPool(ThreadPool&& other) noexcept;

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
        tasks.push([task] { (*task)(); }, queue_capacity);
        return future;
    }

    template <typename T> std::future<T> submit(const std::function<T()>&& function) {
        std::shared_ptr<std::packaged_task<T()>> task =
            std::make_shared<std::packaged_task<T()>>(function);
        std::future<T> future = task->get_future();
        tasks.push([task] { (*task)(); }, queue_capacity);
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
    uint64_t queue_capacity;
    std::vector<std::thread> threads;
    ConcurrentQueue<std::optional<std::function<void()>>> tasks;

    std::barrier<void (*)(void) noexcept> wait_idle_barrier;
};

using ThreadPoolHandle = std::shared_ptr<ThreadPool>;

} // namespace merian
