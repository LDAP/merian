#ifndef _MERIAN_SHADERS_BSDF_GGX_H_
#define _MERIAN_SHADERS_BSDF_GGX_H_

#include "merian-shaders/common.glsl"

// Smiths shadow masking term
float smith_g1(const float roughness, const float wodotn) {
    const float alpha_sq = pow(roughness, 4);
    return 2.0 * wodotn / (wodotn + sqrt(alpha_sq + (1.0 - alpha_sq) * MERIAN_SQUARE(wodotn)));
}

#endif // _MERIAN_SHADERS_BSDF_GGX_H_
