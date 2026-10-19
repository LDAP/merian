#include "merian/vk/command/submission.hpp"

#include <algorithm>
#include <cassert>
#include <exception>
#include <spdlog/spdlog.h>
#include <utility>
#include <vector>

namespace merian {

void Submission::reset(const AsyncRunner& run_async) {
    assert(!cmd && "reset() with recording in progress");
    run_submitted_callbacks(run_async);
    assert(callbacks.empty() && "host callbacks of commands that were never submitted");
    callbacks.clear();
    pool->reset();
}

void Submission::sync_to_cpu(std::function<void()> callback, const CallbackMode mode) {
    cpu_sync_signal_pending = true;
    callbacks.emplace_back(++last_cpu_sync_value, std::move(callback), mode, false);
}

void Submission::sync_to_cpu_and_back(std::function<void()> callback) {
    sync_to_cpu(std::move(callback));
    callbacks.back().signal_back = true;
    submit();
    add_wait_semaphore(cpu_sync_semaphore, vk::PipelineStageFlagBits::eAllCommands,
                       ++last_cpu_sync_value);
}

void Submission::run_finished_callbacks(const AsyncRunner& run_async) {
    if (!next_callback_wait()) {
        return;
    }
    const uint64_t finished = cpu_sync_semaphore->get_counter_value();
    while (!callbacks.empty() && callbacks.front().value <= finished) {
        Callback callback = std::move(callbacks.front());
        callbacks.pop_front();
        if (callback.mode == CallbackMode::ASYNC && run_async) {
            run_async(std::move(callback.function));
            continue;
        }
        try {
            callback.function();
        } catch (const std::exception& e) {
            SPDLOG_ERROR("host callback failed: {}", e.what());
        } catch (...) {
            SPDLOG_ERROR("host callback failed");
        }
        if (callback.signal_back) {
            cpu_sync_semaphore->signal(callback.value + 1);
        }
    }
}

void Submission::run_submitted_callbacks(const AsyncRunner& run_async) {
    while (const auto next = next_callback_wait()) {
        cpu_sync_semaphore->wait(next->second);
        run_finished_callbacks(run_async);
    }
}

std::optional<std::pair<TimelineSemaphoreHandle, uint64_t>> Submission::next_callback_wait() const {
    if (callbacks.empty() || callbacks.front().value > submitted_cpu_sync_value) {
        return std::nullopt;
    }
    return std::make_pair(cpu_sync_semaphore, callbacks.front().value);
}

bool Submission::waits_for_host() const {
    return std::ranges::any_of(callbacks, [&](const Callback& callback) {
        return callback.signal_back && callback.value <= submitted_cpu_sync_value;
    });
}

void Submission::submit() {
    submit_recording(false);
    get_cmd();
}

void Submission::finish() {
    // Submits even if a callback throws: the signals must not stay unsignaled
    std::exception_ptr error;
    for (const auto& callback : std::exchange(pre_submit_callbacks, {})) {
        try {
            callback(*this);
        } catch (...) {
            if (!error) {
                error = std::current_exception();
            }
        }
    }
    submit_recording(true);
    if (error) {
        std::rethrow_exception(error);
    }
}

void Submission::submit_recording(const bool last) {
    std::vector<vk::Semaphore> semaphores;
    std::vector<uint64_t> values;
    if (last) {
        semaphores = std::exchange(signal_semaphores, {});
        values = std::exchange(signal_values, {});
    }
    if (cpu_sync_signal_pending) {
        semaphores.push_back(*cpu_sync_semaphore);
        values.push_back(last_cpu_sync_value);
    }

    get_cmd()->end();
    queue->submit(cmd, VK_NULL_HANDLE, semaphores, wait_semaphores, wait_stages,
                  vk::TimelineSemaphoreSubmitInfo{wait_values, values});
    cmd.reset();
    wait_semaphores.clear();
    wait_stages.clear();
    wait_values.clear();
    if (cpu_sync_signal_pending) {
        submitted_cpu_sync_value = last_cpu_sync_value;
        cpu_sync_signal_pending = false;
    }

    if (last) {
        for (const auto& callback : std::exchange(submit_callbacks, {})) {
            callback(queue, *this);
        }
    }
}

} // namespace merian
