#pragma once

#include "glm/glm.hpp"

namespace merian {

// Reinterprests a float as vec3.
inline glm::vec3* as_vec3(float v[3]) {
    assert(sizeof(glm::vec3) == 3 * sizeof(float));
    return reinterpret_cast<glm::vec3*>(v);
}

}
