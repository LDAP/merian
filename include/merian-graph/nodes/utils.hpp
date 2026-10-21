#pragma once

#include "merian/utils/properties.hpp"

#include <chrono>
#include <cstdint>

namespace merian {

class FrameRate {
  public:
    void properties(Properties& config);

    float get_framerate() const;

    float get_frametime_millis() const;

    // The time of the frame with this index.
    std::chrono::nanoseconds frame_time(const uint64_t frame) const;

  private:
    float framerate = 30;
    float frametime_millis = 1000.f / 30.f;
};

} // namespace merian
