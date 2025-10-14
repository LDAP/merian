#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

namespace merian {

// vectors

using float1 = glm::vec1;
using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;

using int1 = glm::ivec1;
using int2 = glm::ivec2;
using int3 = glm::ivec3;
using int4 = glm::ivec4;

using uint1 = glm::uvec1;
using uint2 = glm::uvec2;
using uint3 = glm::uvec3;
using uint4 = glm::uvec4;

// using half = std::float16_t;
// using half1 = glm::vec<1, std::float16_t, defaultp>;
// using half2 = glm::vec<2, std::float16_t, defaultp>;
// using half3 = glm::vec<3, std::float16_t, defaultp>;
// using half4 = glm::vec<4, std::float16_t, defaultp>;

// matrices

using float1x1 = glm::mat<1, 1, glm::f32, glm::defaultp>;
using float1x2 = glm::mat<1, 2, glm::f32, glm::defaultp>;
using float1x3 = glm::mat<1, 3, glm::f32, glm::defaultp>;
using float1x4 = glm::mat<1, 4, glm::f32, glm::defaultp>;

using float2x1 = glm::mat<2, 1, glm::f32, glm::defaultp>;
using float2x2 = glm::mat<2, 2, glm::f32, glm::defaultp>;
using float2x3 = glm::mat<2, 3, glm::f32, glm::defaultp>;
using float2x4 = glm::mat<2, 4, glm::f32, glm::defaultp>;

using float3x1 = glm::mat<3, 1, glm::f32, glm::defaultp>;
using float3x2 = glm::mat<3, 2, glm::f32, glm::defaultp>;
using float3x3 = glm::mat<3, 3, glm::f32, glm::defaultp>;
using float3x4 = glm::mat<3, 4, glm::f32, glm::defaultp>;

using float4x1 = glm::mat<4, 1, glm::f32, glm::defaultp>;
using float4x2 = glm::mat<4, 2, glm::f32, glm::defaultp>;
using float4x3 = glm::mat<4, 3, glm::f32, glm::defaultp>;
using float4x4 = glm::mat<4, 4, glm::f32, glm::defaultp>;

#define FUNCTION_ALIAS(ALIAS, FUNC)                                                                \
    template <typename... Args>                                                                    \
    auto ALIAS(Args&&... args) -> decltype(FUNC(std::forward<Args>(args)...)) {                    \
        return FUNC(std::forward<Args>(args)...);                                                  \
    }

FUNCTION_ALIAS(mul, glm::operator*)

FUNCTION_ALIAS(normalize, glm::normalize)

FUNCTION_ALIAS(lerp, glm::mix)

FUNCTION_ALIAS(dot, glm::dot)

FUNCTION_ALIAS(cross, glm::cross)

FUNCTION_ALIAS(length, glm::length)

FUNCTION_ALIAS(distance, glm::distance)

FUNCTION_ALIAS(inverse, glm::inverse)

FUNCTION_ALIAS(transpose, glm::transpose)

FUNCTION_ALIAS(acos, glm::acos)

FUNCTION_ALIAS(max, glm::max)

FUNCTION_ALIAS(min, glm::min)

FUNCTION_ALIAS(radians, glm::radians)

FUNCTION_ALIAS(any, glm::any)

FUNCTION_ALIAS(value_ptr, glm::value_ptr)

inline const float1& as_float1(const float f[1]) {
    static_assert(sizeof(float1) == sizeof(float));
    return *reinterpret_cast<const float1*>(f);
}
inline const float2& as_float2(const float f[2]) {
    static_assert(sizeof(float2) == 2 * sizeof(float));
    return *reinterpret_cast<const float2*>(f);
}
inline const float3& as_float3(const float f[3]) {
    static_assert(sizeof(float3) == 3 * sizeof(float));
    return *reinterpret_cast<const float3*>(f);
}
inline const float4& as_float4(const float f[4]) {
    static_assert(sizeof(float4) == 4 * sizeof(float));
    return *reinterpret_cast<const float4*>(f);
}

inline const int1& as_int1(const int32_t f[1]) {
    static_assert(sizeof(int1) == sizeof(int32_t));
    return *reinterpret_cast<const int1*>(f);
}
inline const int2& as_int2(const int32_t f[2]) {
    static_assert(sizeof(int2) == 2 * sizeof(int32_t));
    return *reinterpret_cast<const int2*>(f);
}
inline const int3& as_int3(const int32_t f[3]) {
    static_assert(sizeof(int3) == 3 * sizeof(int32_t));
    return *reinterpret_cast<const int3*>(f);
}
inline const int4& as_int4(const int32_t f[4]) {
    static_assert(sizeof(int4) == 4 * sizeof(int32_t));
    return *reinterpret_cast<const int4*>(f);
}

inline const uint1& as_int1(const uint32_t f[1]) {
    static_assert(sizeof(uint1) == sizeof(uint32_t));
    return *reinterpret_cast<const uint1*>(f);
}
inline const uint2& as_int2(const uint32_t f[2]) {
    static_assert(sizeof(uint2) == 2 * sizeof(uint32_t));
    return *reinterpret_cast<const uint2*>(f);
}
inline const uint3& as_int3(const uint32_t f[3]) {
    static_assert(sizeof(uint3) == 3 * sizeof(uint32_t));
    return *reinterpret_cast<const uint3*>(f);
}
inline const uint4& as_int4(const uint32_t f[4]) {
    static_assert(sizeof(uint4) == 4 * sizeof(uint32_t));
    return *reinterpret_cast<const uint4*>(f);
}

} // namespace merian
