#include "merian/utils/stopwatch.hpp"
#include "merian/utils/chrono.hpp"
#include "merian/utils/string.hpp"
#include <atomic>

namespace merian {

Stopwatch::Stopwatch() {
    reset();
}

void Stopwatch::reset() {
    start = chrono_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

uint64_t Stopwatch::nanos() const {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto end = chrono_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

double Stopwatch::millis() const {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto end = chrono_clock::now();
    return to_milliseconds(end - start);
}

double Stopwatch::seconds() const {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto end = chrono_clock::now();
    return to_seconds(end - start);
}

std::chrono::nanoseconds Stopwatch::duration() const {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto end = chrono_clock::now();
    return end - start;
}

std::ostream& operator<<(std::ostream& stream, const Stopwatch& sw) {
    return stream << format_duration(sw.nanos());
}

auto format_as(const Stopwatch& sw) {
    return format_duration(sw.nanos());
}

} // namespace merian
