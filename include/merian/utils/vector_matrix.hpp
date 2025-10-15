#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <fmt/ranges.h>

namespace merian {

// VECTORS

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

// MATRICES (rows x columns)

// IMPORTANT: glm uses column-major. However we interpret their columns as rows! So we say merian is
// row-major. That also means when the glm construtor says column 1,.. its actually row 1,.. .

template <int ROWS, int COLUMS> using floatRxC = glm::mat<ROWS, COLUMS, glm::f32, glm::defaultp>;

using float1x1 = float1;
using float1x2 = floatRxC<1, 2>;
using float1x3 = floatRxC<1, 3>;
using float1x4 = floatRxC<1, 4>;

using float2x1 = float2;
using float2x2 = floatRxC<2, 2>;
using float2x3 = floatRxC<2, 3>;
using float2x4 = floatRxC<2, 4>;

using float3x1 = float3;
using float3x2 = floatRxC<3, 2>;
using float3x3 = floatRxC<3, 3>;
using float3x4 = floatRxC<3, 4>;

using float4x1 = float4;
using float4x2 = floatRxC<4, 2>;
using float4x3 = floatRxC<4, 3>;
using float4x4 = floatRxC<4, 4>;

template <typename T>
concept GlmType = requires {
    typename std::remove_cvref_t<T>::value_type;
    // These members are present in glm::vec*, glm::mat* etc.
    glm::length_t(std::remove_cvref_t<T>::length());
};

template <typename T>
concept NumericOrGlm = std::is_arithmetic_v<std::remove_cvref_t<T>> || GlmType<T>;

#define FUNCTION_ALIAS(ALIAS, FUNC)                                                                \
    template <typename... Args>                                                                    \
        requires((NumericOrGlm<Args>) && ...)                                                      \
    auto ALIAS(Args&&... args) -> decltype(FUNC(std::forward<Args>(args)...)) {                    \
        return FUNC(std::forward<Args>(args)...);                                                  \
    }

template <int R, int C, int D>
floatRxC<R, D> mul1(const floatRxC<R, C>& m1, const floatRxC<C, D>& m2) {
    return m2 * m1;
}

FUNCTION_ALIAS(mul, glm::operator*)

using glm::normalize;

FUNCTION_ALIAS(lerp, glm::mix)

using glm::dot;

using glm::cross;

using glm::length;

using glm::distance;

using glm::inverse;

using glm::transpose;

using glm::acos;

using glm::max;

using glm::min;

using glm::radians;

using glm::any;

using glm::value_ptr;

template <typename genType = merian::float4x4>
    requires((NumericOrGlm<genType>))
genType identity() {
    return glm::identity<genType>();
}

inline float4x4 look_at(const float3& position, const float3& target, const float3& up) {
    return glm::lookAt(position, target, up);
}

inline float4x4 rotation(const float3& axis, const float angle) {
    return glm::rotate(identity<float4x4>(), angle, axis);
}

inline float4x4 translation(const float3& translation) {
    return glm::translate(identity<float4x4>(), translation);
}

inline float4x4 scale(const float3& scale) {
    return glm::scale(identity<float4x4>(), scale);
}

inline float4x4
perspective(const float fovy, const float aspect, const float near, const float far) {
    return glm::perspective(fovy, aspect, near, far);
}

inline float1 as_float1(const float f[1]) {
    static_assert(sizeof(float1) == sizeof(float));
    float1 v;
    memcpy(&v.x, f, sizeof(float1));
    return v;
}
inline float2 as_float2(const float f[2]) {
    static_assert(sizeof(float2) == 2 * sizeof(float));
    float2 v;
    memcpy(&v.x, f, sizeof(float2));
    return v;
}
inline float3 as_float3(const float f[3]) {
    static_assert(sizeof(float3) == 3 * sizeof(float));
    float3 v;
    memcpy(&v.x, f, sizeof(float3));
    return v;
}
inline float4 as_float4(const float f[4]) {
    static_assert(sizeof(float4) == 4 * sizeof(float));
    float4 v;
    memcpy(&v.x, f, sizeof(float4));
    return v;
}

inline int1 as_int1(const int32_t f[1]) {
    static_assert(sizeof(int1) == sizeof(int32_t));
    int1 v;
    memcpy(&v.x, f, sizeof(int1));
    return v;
}
inline int2 as_int2(const int32_t f[2]) {
    static_assert(sizeof(int2) == 2 * sizeof(int32_t));
    int2 v;
    memcpy(&v.x, f, sizeof(int2));
    return v;
}
inline int3 as_int3(const int32_t f[3]) {
    static_assert(sizeof(int3) == 3 * sizeof(int32_t));
    int3 v;
    memcpy(&v.x, f, sizeof(int3));
    return v;
}
inline int4 as_int4(const int32_t f[4]) {
    static_assert(sizeof(int4) == 4 * sizeof(int32_t));
    int4 v;
    memcpy(&v.x, f, sizeof(int4));
    return v;
}

inline uint1 as_int1(const uint32_t f[1]) {
    static_assert(sizeof(uint1) == sizeof(uint32_t));
    uint1 v;
    memcpy(&v.x, f, sizeof(uint1));
    return v;
}
inline uint2 as_int2(const uint32_t f[2]) {
    static_assert(sizeof(uint2) == 2 * sizeof(uint32_t));
    uint2 v;
    memcpy(&v.x, f, sizeof(uint2));
    return v;
}
inline uint3 as_int3(const uint32_t f[3]) {
    static_assert(sizeof(uint3) == 3 * sizeof(uint32_t));
    uint3 v;
    memcpy(&v.x, f, sizeof(uint3));
    return v;
}
inline uint4 as_int4(const uint32_t f[4]) {
    static_assert(sizeof(uint4) == 4 * sizeof(uint32_t));
    uint4 v;
    memcpy(&v.x, f, sizeof(uint4));
    return v;
}

} // namespace merian

namespace glm {

template <glm::length_t L, typename T, glm::qualifier Q>
inline auto format_as(const glm::vec<L, T, Q>& v) {
    return fmt::format("({})", fmt::join(&v[0], &v[0] + L, ", "));
}

// see above: rows and colums are switched
template <glm::length_t R, glm::length_t C, typename T, glm::qualifier Q>
inline auto format_as(const glm::mat<R, C, T, Q>& m) {
    std::array<std::string, R> rows{};
    for (glm::length_t r = 0; r < R; ++r) {
        rows[r] = fmt::format("{}", m[r]);
    }

    return fmt::format("({})", fmt::join(rows, ",\n "));
}

} // namespace glm
