#pragma once

#include <cmath>
#include <cstdint>
#include <fmt/format.h>
#include <functional>
#include <string>
#include <vector>

namespace merian {

[[nodiscard]] inline std::string format_size(const uint64_t size) {
    std::vector<const char*> units = {"B", "KB", "MB", "GB", "TB"};
    std::size_t unit_index;
    if (size == 0) {
        unit_index = 0;
    } else {
        unit_index = std::min((std::size_t)std::log2(size) / 10, units.size() - 1);
    }
    return fmt::format("{} {}", size / std::pow(1024, unit_index), units[unit_index]);
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
    if (last < value.size())
        function(value.substr(last));
}

} // namespace merian
