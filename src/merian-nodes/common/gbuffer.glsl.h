#include "merian-nodes/common/types.glsl.h"

#ifndef _GBUFFER_H_
#define _GBUFFER_H_

#ifdef __cplusplus
namespace merian {
#else
#extension GL_EXT_shader_explicit_arithmetic_types  : enable
#endif

struct GBuffer {
    uint32_t enc_normal;
    float linear_z;
};

CPP_INLINE uint gbuffer_index(const ivec2 ipos, const ivec2 resolution) {
    return ipos.x + resolution.x * ipos.y;
}

#ifdef __cplusplus
}
#endif

#endif
