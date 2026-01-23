#include "merian/vk/sync/semaphore_timeline.hpp"

#include "merian/vk/utils/check_result.hpp"

namespace merian {

TimelineSemaphore::TimelineSemaphore(const ContextHandle& context, const uint64_t initial_value)
    : Semaphore(context, {vk::SemaphoreType::eTimeline, initial_value}) {}

uint64_t TimelineSemaphore::get_counter_value() const {
    return context->get_device()->get_device().getSemaphoreCounterValue(semaphore);
}

bool TimelineSemaphore::wait(const uint64_t value, const uint64_t timeout_nanos) {
    vk::SemaphoreWaitInfo wait_info{{}, semaphore, value};
    vk::Result result = context->get_device()->get_device().waitSemaphores(wait_info, timeout_nanos);
    if (result == vk::Result::eSuccess)
        return true;
    if (result == vk::Result::eTimeout)
        return false;
    check_result(result, "error waiting on timeline semaphore");
    return false;
}

void TimelineSemaphore::signal(const uint64_t value) {
    vk::SemaphoreSignalInfo signal_info{semaphore, value};
    context->get_device()->get_device().signalSemaphore(signal_info);
}

} // namespace merian
