#include "merian-shaders/common.glsl"
#include "merian-shaders/frames.glsl"

#ifndef _MERIAN_SHADERS_VON_MISES_FISHER_H_
#define _MERIAN_SHADERS_VON_MISES_FISHER_H_

#define MERIAN_SHADERS_VON_MISES_FISHER_IMPL_WENZEL 0
#define MERIAN_SHADERS_VON_MISES_FISHER_IMPL_TOKUYOSHI 1

#define MERIAN_SHADERS_VON_MISES_FISHER_IMPL MERIAN_SHADERS_VON_MISES_FISHER_IMPL_TOKUYOSHI

// ----------------------------------------------------------

#if MERIAN_SHADERS_VON_MISES_FISHER_IMPL == MERIAN_SHADERS_VON_MISES_FISHER_IMPL_WENZEL

// Based on Wenzel Jakob, Numerically stable sampling of the von Mises Fisher distribution on S2 (and other tricks)

float vmf_pdf(const vec3 w, const vec3 mu, const float kappa) {
    if (kappa < 1e-4) return INV_FOUR_PI;
    return kappa / (TWO_PI * (1.0 - exp(-2.0 * kappa))) * exp(kappa * (dot(w, mu) - 1.0));
}

vec3 vmf_sample(const float kappa, const vec2 random) {
    const float w = 1.0 + log(random.x + (1.0 - random.x) * exp(-2.0 * kappa)) / kappa;
    const vec2 v = vec2(sin(TWO_PI * random.y), cos(TWO_PI * random.y));
    return vec3(sqrt(1.0 - w * w) * v, w);
}

#elif MERIAN_SHADERS_VON_MISES_FISHER_IMPL == MERIAN_SHADERS_VON_MISES_FISHER_IMPL_TOKUYOSHI

// Based on Yusuke Tokuyoshi, A Numerically Stable Implementation of the von Mises–Fisher Distribution on S2

float vmf_pdf(const vec3 w, const vec3 mu, const float kappa) {
    const vec3 d = w - mu ;
    return exp (-0.5f * kappa * dot (d, d)) * x_over_expm1(-2.0f * kappa) / FOUR_PI;
}

vec3 vmf_sample(const float kappa, const vec2 random) {
    const float phi = TWO_PI * random.x;
    const float r = kappa > (FLT_EPSILON / 4.0f) ? log1p (random.y * expm1(-2.0f * kappa)) / kappa
                                                 : -2.0f * random.y;
    const float sin_theta = sqrt(-fma(r, r, 2.0f * r));
    return vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, 1.0f + r /* = cos_theta*/);
}

#endif

// ----------------------------------------------------------

// Sample "around" z.
vec3 vmf_sample(const vec3 z, const float kappa, const vec2 random) {
    return make_frame(z) * vmf_sample(kappa, random);
}

// compute concentration parameter for given maximum density x
float vmf_get_kappa(const float pdf) {
    if (pdf > 0.795) return TWO_PI * pdf;
    return max(1e-5, (168.479 * pdf * pdf + 16.4585 * pdf - 2.39942) / (-1.12718 * pdf * pdf + 29.1433 * pdf + 1.0));
}

#endif
