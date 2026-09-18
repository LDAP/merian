#pragma once

#include "merian/utils/enums.hpp"

#include <chrono>
#include <optional>

namespace merian {

MERIAN_ENUM(TimeProviderMode, OFF, WHEN_ACTIVE, ALWAYS)

inline constexpr const char* TIME_PROVIDER_DESCRIPTION =
    "Offer this node as a time source for the graph. While it is selected (the graph's time source "
    "is this node, or Auto and this is the first provider in topological order) the graph clock no "
    "longer follows the wall clock but is the time this node is at. That makes a recording "
    "deterministic and complete regardless of how long a frame actually takes to render.";

// Mixin for Nodes that can drive the graph clock. With no source selected the graph picks the
// first provider in topological order.
class TimeProvider {
  public:
    virtual ~TimeProvider() = default;

    virtual bool provides_time() const {
        return true;
    }

    // The absolute elapsed graph time of the upcoming run, not a step, or nullopt to follow the
    // system clock instead. Called once per run before any pre_process, and only while this
    // provider is the selected time source.
    [[nodiscard]]
    virtual std::optional<std::chrono::nanoseconds> provide_time() = 0;
};

} // namespace merian
