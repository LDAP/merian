#pragma once

#include "glm/glm.hpp"
#include <cstring>

namespace merian {

inline glm::vec4 vec4_from_float(float* v) {
    return glm::vec4(v[0], v[1], v[2], v[3]);
}

inline glm::vec3 vec3_from_float(float* v) {
    return glm::vec3(v[0], v[1], v[2]);
}

inline void copy_to_vec4(float* src, glm::vec4& dst) {
    mempcpy(&dst.x, src, sizeof(float) * 4);
}

inline void copy_to_vec3(float* src, glm::vec3& dst) {
    mempcpy(&dst.x, src, sizeof(float) * 3);
}

} // namespace merian
