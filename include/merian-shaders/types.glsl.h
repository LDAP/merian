#ifndef _MERIAN_SHADERS_TYPES_H_
#define _MERIAN_SHADERS_TYPES_H_

#ifdef __cplusplus

#include "glm/glm.hpp"
#include <cstdint>

using uint = uint32_t;
using vec2 = glm::vec2;
using vec3 = glm::vec3;
using ivec2 = glm::ivec2;
using ivec3 = glm::ivec3;
using vec4 = glm::vec4;
using float16_t = uint16_t;
using uint = uint32_t;
using f16vec2 = float16_t[2];
using f16vec4 = float16_t[4];
using f16vec3 = float16_t[3];
using f16mat3x2 = glm::tmat3x2<uint16_t>;


#define CPP_INLINE inline

#else

#define CPP_INLINE

#endif

#endif
