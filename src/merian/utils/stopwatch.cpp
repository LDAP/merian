#include "merian/utils/stopwatch.hpp"
#include "merian/utils/chrono.hpp"
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

} // namespace merian
