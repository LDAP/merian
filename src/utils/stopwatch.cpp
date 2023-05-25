#include "utils/stopwatch.hpp"

Stopwatch::Stopwatch() {
    reset();
}

void Stopwatch::reset() {
    start = chrono_clock::now();
}

uint64_t Stopwatch::nanos() const {
    auto end = chrono_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

double Stopwatch::millis() const {
    auto end = chrono_clock::now();
    return std::chrono::duration<double, std::nano>(end - start).count();
}

double Stopwatch::seconds() const {
    auto end = chrono_clock::now();
    return std::chrono::duration<double>(end - start).count();
}
