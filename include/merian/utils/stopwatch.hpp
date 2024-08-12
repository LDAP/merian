#pragma once

#include <chrono>

namespace merian {

class Stopwatch {
  private:
    using chrono_clock = std::chrono::high_resolution_clock;

  public:
    /* Creates and starts the stopwatch */
    Stopwatch();
    void reset();
    uint64_t nanos() const;
    double millis() const;
    double seconds() const;
    std::chrono::nanoseconds duration() const;

  private:
    chrono_clock::time_point start;
};

} // namespace merian
