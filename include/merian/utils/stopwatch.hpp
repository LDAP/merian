#pragma once

#include <chrono>
#include <ostream>

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

    friend std::ostream& operator<<(std::ostream& stream, const Stopwatch& sw);

  private:
    chrono_clock::time_point start;
};

auto format_as(const Stopwatch& sw);

} // namespace merian
