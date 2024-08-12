#pragma once

#include <chrono>

namespace merian {

double to_seconds(const auto& chrono_duration) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::chrono::seconds::period>>(
               chrono_duration)
        .count();
}

double to_milliseconds(const auto& chrono_duration) {
    return std::chrono::duration_cast<
               std::chrono::duration<double, std::chrono::milliseconds::period>>(chrono_duration)
        .count();
}

double to_microseconds(const auto& chrono_duration) {
    return std::chrono::duration_cast<
               std::chrono::duration<double, std::chrono::microseconds::period>>(chrono_duration)
        .count();
}

double to_nanoseconds(const auto& chrono_duration) {
    return std::chrono::duration_cast<
               std::chrono::duration<double, std::chrono::nanoseconds::period>>(chrono_duration)
        .count();
}

} // namespace merian
