#pragma once

#include "glm/glm.hpp"

#include <cmath>
#include <cstdint>

namespace merian {

inline uint32_t uint32_from_rgba(float r, float g, float b, float a) {
    return uint32_t(std::round(r * 255)) | (uint32_t(std::round(g * 255)) << 8) |
           uint32_t(std::round(b * 255)) << 16 | uint32_t(std::round(a * 255)) << 24;
};

inline float yuv_luminance(glm::vec3 color) {
    return glm::dot(color, glm::vec3(0.299, 0.587, 0.114));
}

} // namespace merian
