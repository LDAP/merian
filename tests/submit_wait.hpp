#pragma once

#include "merian/vk/command/queue.hpp"
#include "merian/vk/command/submission.hpp"
#include "merian/vk/context.hpp"

#include <functional>

inline void submit_wait(const merian::ContextHandle& context,
                        const merian::QueueHandle& queue,
                        const std::function<void(merian::Submission&)>& record) {
    merian::Submission submission(context, queue);
    record(submission);
    submission.finish();
    queue->wait_idle();
    submission.reset();
}
