#pragma once

#include "glm/glm.hpp"

namespace merian {

// Reinterprests a float as vec3.
inline glm::vec3* as_vec3(float v[3]) {
    assert(sizeof(glm::vec3) == 3 * sizeof(float));
    return reinterpret_cast<glm::vec3*>(v);
}

// Reinterprests a float as vec4.
inline glm::vec4* as_vec4(float v[4]) {
    assert(sizeof(glm::vec4) == 4 * sizeof(float));
    return reinterpret_cast<glm::vec4*>(v);
}

inline const glm::vec3* as_vec3(const float v[3]) {
    assert(sizeof(glm::vec3) == 3 * sizeof(float));
    return reinterpret_cast<const glm::vec3*>(v);
}

// Reinterprests a float as vec4.
inline const glm::vec4* as_vec4(const float v[4]) {
    assert(sizeof(glm::vec4) == 4 * sizeof(float));
    return reinterpret_cast<const glm::vec4*>(v);
}

// Reinterprests a float as uvec3.
inline glm::uvec3* as_uvec3(uint32_t v[3]) {
    assert(sizeof(glm::uvec3) == 3 * sizeof(float));
    return reinterpret_cast<glm::uvec3*>(v);
}

// Reinterprests a float as uvec4.
inline glm::uvec4* as_uvec4(uint32_t v[4]) {
    assert(sizeof(glm::uvec4) == 4 * sizeof(float));
    return reinterpret_cast<glm::uvec4*>(v);
}

} // namespace merian
