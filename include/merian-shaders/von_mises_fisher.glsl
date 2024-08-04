#include "merian-shaders/common.glsl"
#include "merian-shaders/frames.glsl"

#ifndef _VON_MISES_FISHER_H_
#define _VON_MISES_FISHER_H_

// Based on Wenzel Jakob, Numerically stable sampling of the von Mises Fisher distribution on S2 (and other tricks)

// numerically robust von Mises Fisher lobe
float vmf_pdf(const float kappa, const float dotmu) {
    if (kappa < 1e-4) return 1.0 / (4.0 * M_PI);
    return kappa / (2.0 * M_PI * (1.0 - exp(-2.0 * kappa))) * exp(kappa * (dotmu - 1.0));
}

// see wenzel's doc on numerically stable expression for vmm
vec3 vmf_sample(const float kappa, const vec2 random) {
    const float w = 1.0 + log(random.x + (1.0-random.x) * exp(-2.0 * kappa)) / kappa;
    const vec2 v = vec2(sin(2.0 * M_PI * random.y), cos(2.0 * M_PI * random.y));
    return vec3(sqrt(1.0 - w * w) * v, w);
}

// Sample "around" z.
vec3 vmf_sample(const vec3 z, const float kappa, const vec2 random) {
    return make_frame(z) * vmf_sample(kappa, random);
}

// compute concentration parameter for given maximum density x
float vmf_get_kappa(const float pdf) {
    if (pdf > 0.795) return 2.0 * M_PI * pdf;
    return max(1e-5, (168.479 * pdf * pdf + 16.4585 * pdf - 2.39942) / (-1.12718 * pdf * pdf + 29.1433 * pdf + 1.0));
}

#endif
