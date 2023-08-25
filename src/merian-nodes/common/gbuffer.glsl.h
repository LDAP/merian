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

#define gbuffer_index(ipos, resolution) (ipos.x + resolution.x * ipos.y)

#ifdef __cplusplus
}
#endif

#endif
