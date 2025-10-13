#pragma once

#include "hlsl++/matrix_float.h"

#include "hlsl++/vector_float.h"
#include "hlsl++/vector_uint.h"

namespace merian {

// vectors

using float1 = hlslpp::float1;
using float2 = hlslpp::float2;
using float3 = hlslpp::float3;
using float4 = hlslpp::float4;

using int1 = hlslpp::int1;
using int2 = hlslpp::int2;
using int3 = hlslpp::int3;
using int4 = hlslpp::int4;

using uint1 = hlslpp::uint1;
using uint2 = hlslpp::uint2;
using uint3 = hlslpp::uint3;
using uint4 = hlslpp::uint4;

// matrices

using float1x1 = hlslpp::float1x1;
using float1x2 = hlslpp::float1x2;
using float1x3 = hlslpp::float1x3;
using float1x4 = hlslpp::float1x4;

using float2x1 = hlslpp::float2x1;
using float2x2 = hlslpp::float2x2;
using float2x3 = hlslpp::float2x3;
using float2x4 = hlslpp::float2x4;

using float3x1 = hlslpp::float3x1;
using float3x2 = hlslpp::float3x2;
using float3x3 = hlslpp::float3x3;
using float3x4 = hlslpp::float3x4;

using float4x1 = hlslpp::float4x1;
using float4x2 = hlslpp::float4x2;
using float4x3 = hlslpp::float4x3;
using float4x4 = hlslpp::float4x4;

template <typename... Args>
auto mul(Args&&... args) -> decltype(hlslpp::mul(std::forward<Args>(args)...)) {
    return hlslpp::mul(std::forward<Args>(args)...);
}

inline float1 load_float1(float f[1]) {
    float1 v;
    hlslpp::load(v, f);
    return v;
}

inline float2 load_float2(float f[2]) {
    float2 v;
    hlslpp::load(v, f);
    return v;
}

inline float3 load_float3(float f[3]) {
    float3 v;
    hlslpp::load(v, f);
    return v;
}

inline float4 load_float4(float f[4]) {
    float4 v;
    hlslpp::load(v, f);
    return v;
}

template <typename... Args>
auto store(Args&&... args) -> decltype(hlslpp::store(std::forward<Args>(args)...)) {
    return hlslpp::store(std::forward<Args>(args)...);
}

} // namespace merian
