#pragma once

#include <cmath>
#include <cstdint>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace merian {

inline std::string format_size(const uint64_t size) {
    std::vector<const char*> units = {"B", "KB", "MB", "GB", "TB"};
    std::size_t unit_index = std::min((std::size_t) std::log2(size) / 10, units.size() - 1);
    return fmt::format("{} {}", size / std::pow(1024, unit_index), units[unit_index]);
}

} // namespace merian
