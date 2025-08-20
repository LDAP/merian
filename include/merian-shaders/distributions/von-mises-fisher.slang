#pragma once

#include "merian-shaders/utils/math-constants.slangh"

import "distribution.slang";
import "merian-shaders/utils/math.slang";
import "merian-shaders/utils/frame.slang";

namespace merian {

// Based on Wenzel Jakob, Numerically stable sampling of the von Mises Fisher distribution on S2
// (and other tricks)
float vmf_pdf_wenzel(float3 w, float3 mu, float kappa) {
    if (kappa < 1e-4) {
        return MERIAN_INV_FOUR_PI;
    }
    return (kappa / (MERIAN_TWO_PI * (1.0f - exp((-2.0f) * kappa)))) *
           exp(kappa * (dot(w, mu) - 1.0f));
}

float3 vmf_sample_wenzel(float kappa, float2 random) {
    float w = 1.0f + (log(random.x + ((1.0f - random.x) * exp((-2.0f) * kappa))) / kappa);
    float2 v = float2(sin(MERIAN_TWO_PI * random.y), cos(MERIAN_TWO_PI * random.y));
    return float3(v * sqrt(1.0f - (w * w)), w);
}

// --------------------------------

// Based on Yusuke Tokuyoshi, A Numerically Stable Implementation of the von Mises–Fisher
// Distribution on S2
float vmf_pdf_tokuyoshi(float3 w, float3 mu, float kappa) {
    float3 d = w - mu;
    return (exp(((-0.5f) * kappa) * dot(d, d)) * x_over_expm1((-2.0f) * kappa)) / MERIAN_FOUR_PI;
}

float3 vmf_sample_tokuyosh(float kappa, float2 random) {
    float phi = MERIAN_TWO_PI * random.x;
    const float r = kappa > (MERIAN_FLT_EPSILON / 4.0f)
                        ? log1p(random.y * expm1(-2.0f * kappa)) / kappa
                        : -2.0f * random.y;
    float sin_theta = sqrt(-mad(r, r, 2.0f * r));
    return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, 1.0f + r);
}

float3 vmf_sample_tokuyosh(float3 z, float kappa, float2 random) {
    return mul(vmf_sample_tokuyosh(kappa, random), make_frame(z));
}

// --------------------------------

// compute concentration parameter for given maximum density x
float vmf_get_kappa(float max_pdf) {
    if (max_pdf > 0.795) {
        return MERIAN_TWO_PI * max_pdf;
    }
    return max(1e-5, (168.479 * max_pdf * max_pdf + 16.4585 * max_pdf - 2.39942) /
                         (-1.12718 * max_pdf * max_pdf + 29.1433 * max_pdf + 1.0));
}

struct VonMisesFisher3<bool centered> : Distribution<float3> {
    float kappa;
    Conditional<float3, centered> c_mu;

    __init(float kappa, float3 mu) {
        static_assert(!centered, "distribution cannot be centered.");

        this.kappa = kappa;
        this.c_mu = mu;
    }

    __init(float kappa) {
        this.kappa = kappa;
        this.c_mu = float3(0, 0, 1);
    }

    float pdf(const float3 s) {
        if (let mu = c_mu.get()) {
            return vmf_pdf_tokuyoshi(s, mu, kappa);
        } else {
            return vmf_pdf_tokuyoshi(s, float3(0, 0, 1), kappa);
        }
    }

    DistributionSample<float3> sample(inout RandomGenerator rng) {
        if (let mu = c_mu.get()) {
            float3 s = vmf_sample_tokuyosh(mu, kappa, rng.next_float2());
            return DistributionSample<float3>(s, pdf(s));
        } else {
            float3 s = vmf_sample_tokuyosh(kappa, rng.next_float2());
            return DistributionSample<float3>(s, pdf(s));
        }
    }
}

}

