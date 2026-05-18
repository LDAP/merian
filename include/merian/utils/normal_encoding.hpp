#pragma once

#include "merian/utils/vector_matrix.hpp"

#include <cstdint>

namespace merian {

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
uint32_t encode_normal(float* vec) noexcept;

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
uint32_t encode_normal(float3 vec) noexcept;

// Inverse of encode_normal.
float3 decode_normal(uint32_t enc) noexcept;

// Tangent codec: LSB carries handedness sign (0:+1, 1:-1), upper 31 bits the
// oct direction. Matches encode_tangent/decode_tangent in encoding.slang.
uint32_t encode_tangent(float4 t) noexcept;
float4 decode_tangent(uint32_t enc) noexcept;

} // namespace
