#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace merian {

// The first limit reached ends the run.
struct GraphRunOptions {
    // nullopt: no iteration limit; else the nodes see the last one (is_last_iteration)
    std::optional<uint64_t> iterations = 1;
    // comma-separated "node type/identifier/event" patterns, empty parts match any
    std::string until_event{};
    std::optional<std::chrono::nanoseconds> duration{};
    // The graph time of every iteration of the run, over any time source, which continues from it
    // in the next run without a time.
    std::optional<std::chrono::nanoseconds> time{};
};

enum class GraphRunStopReason : uint8_t {
    ITERATIONS,
    EVENT,
    DURATION,
    REQUESTED,
};

struct GraphRunResult {
    GraphRunStopReason stop_reason;
    uint64_t iterations;
    // "node type/identifier/event" of the event that ended the run
    std::string event;
};

} // namespace merian
