#include "common/sampling.glsl"
#include "common/frames.glsl"

#ifndef _BSDF_DIFFUSE_H_
#define _BSDF_DIFFUSE_H_

// n must be normalized
vec3 bsdf_diffuse_sample(const vec3 n, const vec2 random) {
    return make_frame(n) * sample_cos(random);
}

// solid angle
float bsdf_diffuse_pdf(const float wodotn) {
    return INV_PI * wodotn;
}

// solid angle
vec3 bsdf_diffuse_eval(const vec3 albedo) {
    return albedo * INV_PI;
}

#endif
