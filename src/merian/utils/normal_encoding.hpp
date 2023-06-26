#pragma once

#include "glm/glm.hpp"
#include <cstdint>

namespace merian {

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
uint32_t encode_normal(float* vec);

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
uint32_t encode_normal(glm::vec3 vec);

} // namespace
