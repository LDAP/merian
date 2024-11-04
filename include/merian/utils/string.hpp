#pragma once

#include <cmath>
#include <cstdint>
#include <fmt/format.h>
#include <functional>
#include <string>
#include <vector>

namespace merian {

[[nodiscard]] inline std::string format_size(const uint64_t size_bytes) {
    const std::vector<const char*> units = {"B", "KB", "MB", "GB", "TB"};
    std::size_t unit_index;
    if (size_bytes == 0) {
        unit_index = 0;
    } else {
        unit_index = std::min((std::size_t)std::log2(size_bytes) / 10, units.size() - 1);
    }
    return fmt::format("{} {}", (double)size_bytes / std::pow(1024, unit_index), units[unit_index]);
}

[[nodiscard]] inline std::string format_duration(const uint64_t duration_ns) {
    const std::vector<const char*> units = {"ns", "μs", "ms", "s"};
    std::size_t unit_index;
    if (duration_ns == 0) {
        unit_index = 0;
    } else {
        unit_index = std::min((std::size_t)(std::log10(duration_ns) / 3), units.size() - 1);
    }
    return fmt::format("{} {}", (double)duration_ns / std::pow(1000, unit_index), units[unit_index]);
}

[[nodiscard]] inline bool ends_with(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size())
        return false;
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

[[nodiscard]] inline bool starts_with(const std::string& value, const std::string& prefix) {
    if (prefix.size() > value.size())
        return false;
    return std::equal(prefix.begin(), prefix.end(), value.begin());
}

inline void split(const std::string& value,
                  const std::string& delim,
                  const std::function<void(const std::string&)> function) {
    std::size_t last = 0;
    std::size_t pos = 0;
    while ((pos = value.find(delim, last)) != std::string::npos) {
        function(value.substr(last, pos - last));
        last = pos + delim.size();
    }
    function(value.substr(last));
}

} // namespace merian
