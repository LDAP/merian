#include "merian-shaders/types.glsl.h"

#ifndef _MERIAN_SHADERS_GBUFFER_H_
#define _MERIAN_SHADERS_GBUFFER_H_

#ifdef __cplusplus
namespace merian_nodes {
#else
#extension GL_EXT_shader_explicit_arithmetic_types  : enable
#extension GL_EXT_scalar_block_layout : enable
#endif

struct GBuffer {
    // encoded normal of pixel
    uint32_t enc_normal;
    // linear distance from camera to pixel
    float linear_z;
    // dlinear_z / dipos in depth / pixel
    f16vec2 grad_z;
    // camera velocity in ray direction
    float vel_z;
};

// power of two
#define gbuffer_block_size_power 3 // 2^3 = 8
#define gbuffer_block_size (1 << gbuffer_block_size_power)
#define gbuffer_block_size_minus_one (gbuffer_block_size - 1)
// increases the number such that it is divisible by the block size
#define gbuffer_dimension_for_block_size(number) (((number) + gbuffer_block_size_minus_one) & ~gbuffer_block_size_minus_one)
// computes the buffer size for the GBuffer
#define gbuffer_size(width, height) (gbuffer_dimension_for_block_size(width) * gbuffer_dimension_for_block_size(height))
// only valid in C
#define gbuffer_size_bytes(width, height) (gbuffer_size(width, height) * sizeof(merian_nodes::GBuffer))

// z-Curve for better memory locality
#define gbuffer_block(ipos, resolution) (((ipos).x >> gbuffer_block_size_power) + (((resolution).x + gbuffer_block_size_minus_one) >> gbuffer_block_size_power) * ((ipos).y >> gbuffer_block_size_power))
#define gbuffer_inner(ipos) (((ipos).x & gbuffer_block_size_minus_one) + gbuffer_block_size * ((ipos).y & gbuffer_block_size_minus_one))
#define gbuffer_index(ipos, resolution) (gbuffer_inner(ipos) + gbuffer_block(ipos, resolution) * gbuffer_block_size * gbuffer_block_size)

#ifdef __cplusplus
}
#endif

#endif
