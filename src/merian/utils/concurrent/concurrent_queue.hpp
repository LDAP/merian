#pragma once

#include <condition_variable>
#include <mutex>
#include <vector>

namespace merian {

template <typename T> class ConcurrentQueue {
  public:
    ConcurrentQueue() {}

    ~ConcurrentQueue() {}

    ConcurrentQueue(ConcurrentQueue& other) = delete;

    ConcurrentQueue(ConcurrentQueue&& other) {
        std::unique_lock lk_other(other.mutex);
        std::unique_lock lk(mutex);
        q = std::move(other.q);
    }

    ConcurrentQueue& operator=(const ConcurrentQueue& src) = delete;

    ConcurrentQueue& operator=(ConcurrentQueue&& src) {
        if (this == &src)
            return *this;
        this->~ConcurrentQueue();
        new (this) ConcurrentQueue(std::move(src));
        return *this;
    }

    void push(const T&& value, const uint64_t max_size = UINT64_MAX) {
        std::unique_lock lk(mutex);
        cv_full.wait(lk, [&] { return q.size() < max_size; });
        q.emplace_back(std::move(value));
        lk.unlock();
        cv_empty.notify_all();
    }

    void push(const T& value, const uint64_t max_size = UINT64_MAX) {
        std::unique_lock lk(mutex);
        cv_full.wait(lk, [&] { return q.size() < max_size; });
        q.push_back(value);
        lk.unlock();
        cv_empty.notify_all();
    }

    bool empty() {
        std::lock_guard lk(mutex);
        return q.empty();
    }

    void wait_empty() {
        std::unique_lock lk(mutex);
        cv_empty.wait(lk, [&] { return q.empty(); });
        return;
    }

    T pop() {
        std::unique_lock lk(mutex);
        cv_empty.wait(lk, [&] { return !q.empty(); });
        T t = std::move(q.back());
        q.pop_back();
        lk.unlock();
        cv_full.notify_all();
        return std::move(t);
    }

  private:
    std::vector<T> q;
    std::condition_variable cv_empty;
    std::condition_variable cv_full;
    std::mutex mutex;
};

} // namespace merian
