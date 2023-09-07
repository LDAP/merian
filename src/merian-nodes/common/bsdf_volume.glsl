#include "common/sampling.glsl"
#include "common/frames.glsl"

#ifndef _BSDF_VOLUME_H_
#define _BSDF_VOLUME_H_

vec3 bsdf_volume_isotropic_sample(const vec2 random) {
    return sample_sphere(random);
}

// solid angle
float bsdf_volume_isotropic_pdf() {
    return INV_PI / 4;
}

// solid angle
vec3 bsdf_volume_isotropic_eval() {
    return vec3(INV_PI / 4);
}

#endif
