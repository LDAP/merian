#include "merian-graph/nodes/host_sink/host_sink.hpp"

#include <algorithm>
#include <utility>

namespace merian {

bool HostSink::Capture::ready() const {
    return semaphore->get_counter_value() >= semaphore_value;
}

HostSink::NodeStatusFlags
HostSink::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    if (!capturable(io) || !consume_request(info)) {
        return {};
    }
    const HostCopy copy = cmd_capture(submission.get_cmd(), io, info);
    {
        const std::scoped_lock lock(mutex);
        captures.push_back({
            .copy = copy,
            .run_iteration = info.get_iteration(),
            .time = info.get_elapsed(),
            .semaphore = info.get_iteration_semaphore(),
            .semaphore_value = info.get_total_iteration() + 1,
        });
    }
    captured.notify_all();
    return {};
}

void HostSink::capture(const uint32_t count) {
    const std::scoped_lock lock(mutex);
    requested_count += count;
}

void HostSink::capture_at(const std::vector<uint64_t>& iterations) {
    const std::scoped_lock lock(mutex);
    requested_iterations.insert(iterations.begin(), iterations.end());
}

void HostSink::capture_every(const uint32_t every) {
    const std::scoped_lock lock(mutex);
    this->every = every;
}

std::optional<HostSink::Capture>
HostSink::get(const std::optional<std::chrono::nanoseconds>& timeout) {
    const auto deadline =
        std::chrono::steady_clock::now() + timeout.value_or(std::chrono::nanoseconds::max() / 2);
    std::unique_lock lock(mutex);
    while (captured.wait_until(lock, deadline, [&] { return !captures.empty() || cancelled; }) &&
           !captures.empty()) {
        const Capture oldest = captures.front();
        lock.unlock();
        const auto remaining = std::max(deadline - std::chrono::steady_clock::now(),
                                        std::chrono::steady_clock::duration::zero());
        const bool ready = oldest.semaphore->wait(
            oldest.semaphore_value,
            std::chrono::duration_cast<std::chrono::nanoseconds>(remaining).count());
        lock.lock();
        if (!ready) {
            return std::nullopt;
        }
        // another consumer may have taken it meanwhile
        if (!captures.empty() && captures.front().semaphore_value == oldest.semaphore_value) {
            captures.pop_front();
            return oldest;
        }
    }
    return std::nullopt;
}

std::vector<HostSink::Capture> HostSink::get_all() {
    const std::scoped_lock lock(mutex);
    std::vector<Capture> ready;
    while (!captures.empty() && captures.front().ready()) {
        ready.push_back(std::move(captures.front()));
        captures.pop_front();
    }
    return ready;
}

std::optional<uint64_t> HostSink::next_ready_semaphore_value() const {
    const std::scoped_lock lock(mutex);
    if (captures.empty() || !captures.front().ready()) {
        return std::nullopt;
    }
    return captures.front().semaphore_value;
}

void HostSink::cancel() {
    {
        const std::scoped_lock lock(mutex);
        requested_count = 0;
        requested_iterations.clear();
        every = 0;
        cancelled = true;
    }
    captured.notify_all();
}

void HostSink::on_removed() {
    cancel();
}

bool HostSink::is_cancelled() const {
    const std::lock_guard lock(mutex);
    return cancelled;
}

bool HostSink::capturable([[maybe_unused]] const NodeIO& io) const {
    return true;
}

void HostSink::capture_properties(Properties& config) const {
    // unlocked while recording: the recorder may wait for a host thread that waits for the lock
    const auto [waiting, skipped_count] = [&] {
        const std::scoped_lock lock(mutex);
        return std::pair{captures.size(), skipped};
    }();
    config.output_text("waiting: {}, skipped: {}", waiting, skipped_count);
}

bool HostSink::consume_request(const NodeProcessInfo& info) {
    const uint64_t iteration = info.get_iteration();
    const std::scoped_lock lock(mutex);
    requested_iterations.erase(requested_iterations.begin(),
                               requested_iterations.lower_bound(iteration));
    if (requested_iterations.erase(iteration) != 0) {
        return true;
    }
    if (requested_count > 0) {
        requested_count--;
        return true;
    }
    if (every == 0 || iteration % every != 0) {
        return false;
    }
    if (captures.size() > info.get_iterations_in_flight()) {
        skipped++;
        return false;
    }
    return true;
}

} // namespace merian
