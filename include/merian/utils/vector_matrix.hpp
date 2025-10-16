#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <fmt/ranges.h>

namespace merian {

// VECTORS

template <int L, typename T> using vecL = glm::vec<L, T, glm::defaultp>;

using float1 = vecL<1, float>;
using float2 = vecL<2, float>;
using float3 = vecL<3, float>;
using float4 = vecL<4, float>;

using int1 = vecL<1, std::int32_t>;
using int2 = vecL<2, std::int32_t>;
using int3 = vecL<3, std::int32_t>;
using int4 = vecL<4, std::int32_t>;

using uint1 = vecL<1, std::uint32_t>;
using uint2 = vecL<2, std::uint32_t>;
using uint3 = vecL<3, std::uint32_t>;
using uint4 = vecL<4, std::uint32_t>;

// using half = std::float16_t;
// using half1 = glm::vec<1, std::float16_t, defaultp>;
// using half2 = glm::vec<2, std::float16_t, defaultp>;
// using half3 = glm::vec<3, std::float16_t, defaultp>;
// using half4 = glm::vec<4, std::float16_t, defaultp>;

// MATRICES (rows x columns)

// IMPORTANT: glm uses column-major. However we interpret their columns as rows! So we say merian is
// row-major. That also means when the glm construtor says column 1,.. its actually row 1,.. .

template <int ROWS, int COLUMS, typename T> using matRxC = glm::mat<ROWS, COLUMS, T, glm::defaultp>;
template <int ROWS, int COLUMS> using floatRxC = matRxC<ROWS, COLUMS, float>;

using float1x1 = float1;
using float1x2 = float2;
using float1x3 = float3;
using float1x4 = float4;

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

template <int R, int C, int D, typename T>
matRxC<R, D, T> constexpr mul(const matRxC<R, C, T>& m1, const matRxC<C, D, T>& m2) {
    // see above: glm is column-major so we need to adapt the operations to our row-major view.
    return m2 * m1;
}

template <int R, int C, typename T>
vecL<R, T> constexpr mul(const matRxC<R, C, T>& m, const vecL<C, T>& v) {
    // see above: glm is column-major so we need to adapt the operations to our row-major view.
    return v * m;
}

template <int R, int C, typename T>
vecL<C, T> constexpr mul(const vecL<R, T>& v, const matRxC<R, C, T>& m) {
    // see above: glm is column-major so we need to adapt the operations to our row-major view.
    return m * v;
}

using glm::normalize;

FUNCTION_ALIAS(lerp, glm::mix)

using glm::dot;

using glm::cross;

using glm::length;

using glm::distance;

using glm::inverse;

using glm::acos;

using glm::max;

using glm::min;

using glm::radians;

using glm::any;

using glm::value_ptr;

template <typename genType = merian::float4x4>
    requires((NumericOrGlm<genType>))
constexpr genType identity() {
    return glm::identity<genType>();
}

template <typename genType = merian::float4x4>
    requires((NumericOrGlm<genType>))
constexpr genType zeros() {
    return glm::zero<genType>();
}

inline float4x4 rotation(const float3& axis, const float angle) {
    const float a = angle;
    const float c = std::cos(a);
    const float s = std::sin(a);

    float3 n_axis(normalize(axis));
    float3 temp((1.f - c) * n_axis);

    float4x4 rot = identity();
    rot[0][0] = c + (temp[0] * n_axis[0]);
    rot[0][1] = (temp[1] * n_axis[0]) - (s * n_axis[2]);
    rot[0][2] = (temp[2] * n_axis[0]) + (s * n_axis[1]);
    rot[1][0] = (temp[0] * n_axis[1]) + (s * n_axis[2]);
    rot[1][1] = c + (temp[1] * n_axis[1]);
    rot[1][2] = (temp[2] * n_axis[1]) - (s * n_axis[0]);
    rot[2][0] = (temp[0] * n_axis[2]) - (s * n_axis[1]);
    rot[2][1] = (temp[1] * n_axis[2]) + (s * n_axis[0]);
    rot[2][2] = c + (temp[2] * n_axis[2]);
    return rot;
}

template <int R, int C, typename T> constexpr matRxC<C, R, T> transpose(const matRxC<R, C, T>& m) {
    matRxC<C, R, T> result;
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < R; j++) {
            result[i][j] = m[j][i];
        }
    }
    return result;
}

template <int L, typename T> constexpr vecL<L, T> transpose(const vecL<L, T>& v) {
    // we do not distinguish between rows and columns.
    return v;
}

constexpr float4x4 translation(const float3& translation) {
    float4x4 m = identity();
    m[0][3] = translation[0];
    m[1][3] = translation[1];
    m[2][3] = translation[2];
    return m;
}

constexpr float4x4 scale(const float3& scale) {
    return float4x4(scale.x, 0, 0, 0, 0, scale.y, 0, 0, 0, 0, scale.z, 0, 0, 0, 0, 1);
}

inline float4x4 look_at(const float3& position, const float3& target, const float3& up) {
    return transpose(glm::lookAt(position, target, up));
}

inline float4x4
perspective(const float fovy, const float aspect, const float near, const float far) {
    return transpose(glm::perspective(fovy, aspect, near, far));
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
constexpr auto format_as(const glm::vec<L, T, Q>& v) {
    return fmt::format("({})", fmt::join(&v[0], &v[0] + L, ", "));
}

// see above: rows and colums are switched
template <glm::length_t R, glm::length_t C, typename T, glm::qualifier Q>
constexpr auto format_as(const glm::mat<R, C, T, Q>& m) {
    std::array<std::string, R> rows{};
    for (glm::length_t r = 0; r < R; ++r) {
        rows[r] = fmt::format("{}", m[r]);
    }

    return fmt::format("({})", fmt::join(rows, ",\n "));
}

} // namespace glm

namespace {
template <int R1, int C1, int R2, int C2, typename T>
inline auto operator*(const merian::matRxC<R1, C1, T>& m1, const merian::matRxC<R2, C2, T>& m2) {
    static_assert(false, "use merian::mul(..) instead!");
    return m1 * m2;
}

template <int R, int C, int L, typename T>
inline merian::vecL<R, T> operator*(const merian::matRxC<R, C, T>& m, const merian::vecL<L, T>& v) {
    static_assert(false, "use merian::mul(..) instead!");
    return m * v;
}

template <int R, int C, int L, typename T>
inline merian::vecL<C, T> operator*(const merian::vecL<L, T>& v, const merian::matRxC<R, C, T>& m) {
    static_assert(false, "use merian::mul(..) instead!");
    return v * m;
}

} // namespace
