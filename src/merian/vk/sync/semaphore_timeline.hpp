#pragma once

#include "merian/vk/sync/semaphore.hpp"

namespace merian {
class TimelineSemaphore : public Semaphore {
  public:
    TimelineSemaphore(const SharedContext& context, const uint64_t initial_value = 0)
        : Semaphore(context, {vk::SemaphoreType::eTimeline, initial_value}) {}

    uint64_t get_counter_value() const {
        return context->device.getSemaphoreCounterValue(semaphore);
    }

    // Waits until the semaphore holds a value that is >= the supplied value.
    // If timeout_nanos > 0: returns true of the value was signaled, false if the timeout was
    // reached. If timeout_nanos = 0: returns true if the value was signaled, false otherwise (does
    // not wait).
    bool wait(const uint64_t value, const uint64_t timeout_nanos = UINT64_MAX) {
        vk::SemaphoreWaitInfo wait_info{{}, semaphore, value};
        vk::Result result = context->device.waitSemaphores(wait_info, timeout_nanos);
        if (result == vk::Result::eSuccess)
            return true;
        if (result == vk::Result::eTimeout)
            return false;
        check_result(result, "error waiting on timeline semaphore");
        return false;
    }

    void signal(const uint64_t value) {
        vk::SemaphoreSignalInfo signal_info{semaphore, value};
        context->device.signalSemaphore(signal_info);
    }
};

using TimelineSemaphoreHandle = std::shared_ptr<TimelineSemaphore>;
} // namespace merian