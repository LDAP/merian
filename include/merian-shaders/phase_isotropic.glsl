
#extension GL_GOOGLE_include_directive : require

#include "sampling.glsl"
#include "frames.glsl"

#ifndef _MERIAN_SHADERS_PHASE_ISOTROPIC_H_
#define _MERIAN_SHADERS_PHASE_ISOTROPIC_H_

vec3 phase_isotropic_sample(vec2 random) {
    return sample_sphere(random);
}

// solid angle
float phase_isotropic_pdf() {
    return (INV_PI / 4);
}

// solid angle
vec3 phase_isotropic_eval() {
    return vec3(INV_PI / 4);
}

#endif
