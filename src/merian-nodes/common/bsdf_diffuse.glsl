#include "common/sampling.glsl"
#include "common/frames.glsl"

#ifndef _BSDF_DIFFUSE_H_
#define _BSDF_DIFFUSE_H_

// n must be normalized
vec3 bsdf_diffuse_sample(const vec3 n, const vec2 random) {
    return make_frame(n) * sample_cos(random);
}

float bsdf_diffuse_pdf() {
    return 1.0 / M_PI;
}

float bsdf_diffuse_eval() {
    return 1.0 / M_PI;
}

#endif
