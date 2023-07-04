#include "common/sampling.glsl"

#ifndef _BSDF_DIFFUSE_H_
#define _BSDF_DIFFUSE_H_

// du, dv, n define the coordinate system in which to sample
vec3 bsdf_diffuse_sample(const vec3 wi, const vec3 du, const vec3 dv, const vec3 n, const vec2 random) {
    return mat3(du, dv, n) * sample_cos(random);
}

float bsdf_diffuse_pdf() {
    return 1.0 / M_PI;
}

float bsdf_diffuse_eval() {
    return 1.0 / M_PI;
}

#endif
