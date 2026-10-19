#pragma once

#include "merian-graph/graph/node.hpp"
#include "merian/vk/sync/semaphore_timeline.hpp"
#include "merian/vk/utils/host_copy.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <set>
#include <vector>

namespace merian {

// Copies its input to host memory on request. The host methods are thread-safe.
class HostSink : public Node {
  public:
    struct Capture {
        HostCopy copy;
        uint64_t run_iteration;
        // graph time in seconds
        double time;
        TimelineSemaphoreHandle semaphore;
        uint64_t semaphore_value;

        bool ready() const;
    };

    NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    void on_removed() override;

    void capture(const uint32_t count = 1);

    // Past iterations are ignored.
    void capture_at(const std::vector<uint64_t>& iterations);

    // 0 stops; skips a capture while more captures wait than iterations are in flight.
    void capture_every(const uint32_t every);

    // Oldest capture, waiting up to timeout (forever if unset); nullopt on timeout or cancel.
    std::optional<Capture> get(const std::optional<std::chrono::nanoseconds>& timeout = {});

    // The ready captures, oldest first, without waiting.
    std::vector<Capture> get_all();

    std::optional<uint64_t> next_ready_semaphore_value() const;

    // Drops outstanding requests; get returns nullopt once no capture waits.
    void cancel();

    bool is_cancelled() const;

  protected:
    virtual bool capturable(const NodeIO& io) const;

    virtual HostCopy
    cmd_capture(const CommandBufferHandle& cmd, const NodeIO& io, const NodeProcessInfo& info) = 0;

    void capture_properties(Properties& config) const;

  private:
    bool consume_request(const NodeProcessInfo& info);

    mutable std::mutex mutex;
    std::condition_variable captured;
    std::deque<Capture> captures;
    uint32_t requested_count = 0;
    std::set<uint64_t> requested_iterations;
    uint32_t every = 0;
    uint64_t skipped = 0;
    bool cancelled = false;
};

} // namespace merian
