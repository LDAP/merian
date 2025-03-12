#pragma once

#include "merian/vk/sync/semaphore.hpp"

namespace merian {

class TimelineSemaphore;
using TimelineSemaphoreHandle = std::shared_ptr<TimelineSemaphore>;

class TimelineSemaphore : public Semaphore {
  private:
    TimelineSemaphore(const ContextHandle& context, const uint64_t initial_value = 0);

  public:
    uint64_t get_counter_value() const;

    // Waits until the semaphore holds a value that is >= the supplied value.
    // If timeout_nanos > 0: returns true of the value was signaled, false if the timeout was
    // reached. If timeout_nanos = 0: returns true if the value was signaled, false otherwise (does
    // not wait).
    bool wait(const uint64_t value, const uint64_t timeout_nanos = UINT64_MAX);

    void signal(const uint64_t value);

  public:
    static TimelineSemaphoreHandle create(const ContextHandle& context,
                                          const uint64_t initial_value = 0) {
        return std::shared_ptr<TimelineSemaphore>(new TimelineSemaphore(context, initial_value));
    }
};

} // namespace merian
