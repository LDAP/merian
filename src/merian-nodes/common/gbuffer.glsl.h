#include "merian-nodes/common/types.glsl.h"

#ifndef _GBUFFER_H_
#define _GBUFFER_H_

#ifdef __cplusplus
namespace merian_nodes {
#else
#extension GL_EXT_shader_explicit_arithmetic_types  : enable
#endif

struct GBuffer {
    // encoded normal of pixel
    uint32_t enc_normal;
    // linear distance from camera to pixel
    float linear_z;
    // dlinear_z / dipos in depth / pixel
    vec2 grad_z;
    // camera velocity in ray direction
    float vel_z;
};

// Defines the block size in which a morton curve is used.
#define gbuffer_block_size_power 5 // 2^5 = 32
#define gbuffer_block_size (1 << gbuffer_block_size_power)
#define gbuffer_block_size_minus_one (gbuffer_block_size - 1)


// Round up to a multiple of block size
#define gbuffer_dimension_for_block_size(number) ((number + gbuffer_block_size_minus_one) & ~gbuffer_block_size_minus_one)
// computes the buffer size for the GBuffer
#define gbuffer_size(width, height) (gbuffer_dimension_for_block_size(width) * gbuffer_dimension_for_block_size(height))
// only valid in C
#define gbuffer_size_bytes(width, height) (gbuffer_size(width, height) * sizeof(merian_nodes::GBuffer))

#ifndef __cplusplus

#include "common/morton.glsl"

// z-Curve for better memory locality
#define gbuffer_block(ipos, resolution) ((ipos.x >> gbuffer_block_size_power) + ((resolution.x + gbuffer_block_size_minus_one) >> gbuffer_block_size_power) * (ipos.y >> gbuffer_block_size_power))
#define gbuffer_inner(ipos) ((ipos.x & gbuffer_block_size_minus_one) + gbuffer_block_size * (ipos.y & gbuffer_block_size_minus_one))
#define gbuffer_index(ipos, resolution) (morton_encode2d(ipos & gbuffer_block_size_minus_one) + gbuffer_block(ipos, resolution) * gbuffer_block_size * gbuffer_block_size)

#endif

#ifdef __cplusplus
}
#endif

#endif
