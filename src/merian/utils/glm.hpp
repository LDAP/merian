#pragma once

#include "glm/glm.hpp"

namespace merian {

// Reinterprests a float as vec3.
inline glm::vec3* as_vec3(float v[3]) {
    assert(sizeof(glm::vec3) == 3 * sizeof(float));
    return reinterpret_cast<glm::vec3*>(v);
}

// Reinterprests a float as vec3.
inline glm::vec4* as_vec4(float v[4]) {
    assert(sizeof(glm::vec4) == 4 * sizeof(float));
    return reinterpret_cast<glm::vec4*>(v);
}

}
