#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace merian {

template <typename T> class ConcurrentQueue {
  public:
    ConcurrentQueue() {}

    ~ConcurrentQueue() {}

    ConcurrentQueue(ConcurrentQueue& other) = delete;

    ConcurrentQueue(ConcurrentQueue&& other)  noexcept {
        std::unique_lock lk_other(other.mutex);
        std::unique_lock lk(mutex);
        q = std::move(other.q);
    }

    ConcurrentQueue& operator=(const ConcurrentQueue& src) = delete;

    ConcurrentQueue& operator=(ConcurrentQueue&& src)  noexcept {
        if (this == &src)
            return *this;
        this->~ConcurrentQueue();
        new (this) ConcurrentQueue(std::move(src));
        return *this;
    }

    void push(const T&& value, const uint64_t max_size = UINT64_MAX) {
        std::unique_lock lk(mutex);
        cv_full.wait(lk, [&] { return q.size() < max_size; });
        q.push(value);
        lk.unlock();
        cv_empty.notify_all();
    }

    void push(const T& value, const uint64_t max_size = UINT64_MAX) {
        std::unique_lock lk(mutex);
        cv_full.wait(lk, [&] { return q.size() < max_size; });
        q.push(value);
        lk.unlock();
        cv_empty.notify_all();
    }

    std::size_t size() {
        std::unique_lock lk(mutex);
        return q.size();
    }

    bool empty() {
        std::lock_guard lk(mutex);
        return q.empty();
    }

    void wait_empty() {
        std::unique_lock lk(mutex);
        cv_full.wait(lk, [&] { return q.empty(); });
    }

    T pop() {
        std::unique_lock lk(mutex);
        cv_empty.wait(lk, [&] { return !q.empty(); });
        T t = std::move(q.front());
        q.pop();
        lk.unlock();
        cv_full.notify_all();
        return t;
    }

  private:
    std::queue<T> q;
    std::condition_variable cv_empty;
    std::condition_variable cv_full;
    std::mutex mutex;
};

} // namespace merian
