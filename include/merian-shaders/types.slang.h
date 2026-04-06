#ifndef _MERIAN_SHADERS_TYPES_SLANG_H_
#define _MERIAN_SHADERS_TYPES_SLANG_H_

#ifdef __cplusplus

#include "merian/utils/vector_matrix.hpp"

#include <cstdint>

using float2 = merian::float2;
using float3 = merian::float3;
using float4 = merian::float4;
using int2 = merian::int2;
using int3 = merian::int3;
using uint2 = merian::uint2;
using uint3 = merian::uint3;

// half2 mapped to packed representation (two float16 values)
struct half2 {
    uint16_t x;
    uint16_t y;
};

#endif

#endif
