#pragma once

#include <cmath>
#include <cstdint>

namespace merian {

inline uint32_t uint32_from_rgba(float r, float g, float b, float a) {
    return uint32_t(std::round(r * 255)) | (uint32_t(std::round(g * 255)) << 8) |
           uint32_t(std::round(b * 255)) << 16 | uint32_t(std::round(a * 255)) << 24;
};

} // namespace merian
