#ifndef _MERIAN_SHADERS_TYPES_H_
#define _MERIAN_SHADERS_TYPES_H_

#ifdef __cplusplus

#include "merian/utils/vector_matrix.hpp"

#include "hlsl++/half.h"
#include <cstdint>

using uint = uint32_t;
using vec2 = merian::float2;
using vec3 = merian::float3;
using ivec2 = merian::int2;
using ivec3 = merian::int3;
using vec4 = merian::float4;
using float16_t = merian::half;
using uint = uint32_t;
using f16vec2 = float16_t[2];
using f16vec4 = float16_t[4];
using f16vec3 = float16_t[3];
using f16mat3x2 = float16_t[6];


#define CPP_INLINE inline

#else

#define CPP_INLINE

#endif

#endif
