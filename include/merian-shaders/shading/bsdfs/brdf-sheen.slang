#include "merian-shaders/utils/math-constants.slangh"

import bsdf;
import merian_shaders.utils.sampling;

namespace merian {

// Microfacet sheen BRDF (Estevez & Kulla 2017).
struct SheenBRDF : BSDF {
    float3 color;
    // Sheen uses the perceptual roughness directly as the distribution width (no GGX alpha = r^2
    // remap), per Estevez & Kulla.
    float roughness;

    __init(float3 color, float roughness) {
        this.color = color;
        this.roughness = max(roughness, 1e-6f);
    }

    float sheen_D(const float ndoth) {
        const float inv_alpha = 1.f / roughness;
        const float cos2h = ndoth * ndoth;
        const float sin2h = max(1.f - cos2h, 0.0078125f);
        return (2.f + inv_alpha) * pow(sin2h, inv_alpha * 0.5f) * (0.5f * MERIAN_INV_PI);
    }

    // Rational fit for the visibility lambda term, parameterised by (1 - roughness)^2.
    float sheen_L(const float x) {
        const float r = (1.f - roughness) * (1.f - roughness);
        const float inv_r = 1.f - r;
        const float a = 25.3245f * r + 21.5473f * inv_r;
        const float b = 3.32435f * r + 3.82987f * inv_r;
        const float c = 0.16801f * r + 0.19823f * inv_r;
        const float d = -1.27393f * r - 1.97760f * inv_r;
        const float e = -4.85967f * r - 4.32054f * inv_r;
        return a / (1.f + b * pow(x, c)) + d * x + e;
    }

    float sheen_lambda(const float c) {
        if (c < 0.5f) {
            return exp(sheen_L(c));
        }
        return exp(2.f * sheen_L(0.5f) - sheen_L(1.f - c));
    }

    float sheen_G(const float ndoto, const float ndoti) {
        // Terminator softening (Estevez & Kulla eq. 4): raise the outgoing lambda.
        return 1.f / (1.f + pow(sheen_lambda(ndoto), 1.f + 2.f * pow(1.f - ndoto, 8.f)) +
                      sheen_lambda(ndoti));
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return float3(0.f);
        }
        const float3 h = normalize(wi + wo);
        // eval returns f * wo.z, so one foreshortening cosine cancels.
        return sheen_D(h.z) * sheen_G(wo.z, wi.z) * color / (4.f * wi.z);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        const float3 wo = sample_cos(rng);
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return none;
        }
        return wo;
    }

    float pdf(const float3 wi, const float3 wo) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return 0.f;
        }
        return MERIAN_INV_PI * wo.z;
    }

    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        if (let s = sample_eval_impl<This>(this, wi, rng)) {
            ShadingFunctionSample r = s;
            r.lobe = BSDFLobe::Diffuse;
            return r;
        }
        return none;
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(color * sheen_directional_albedo(wi.z, roughness));
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return float3(0.f);
    }
}

// Sheen directional albedo (hemispherical reflectance) of the Estevez & Kulla BRDF, tabulated per
// roughness (0..1 in 0.1 steps) and fit in cosTheta; the stored curve is the transmitted fraction,
// so the reflectance is 1 - it. Coefficients from Falcor's SheenBSDF.
float sheen_directional_albedo(const float ndotv, const float roughness) {
    const float coeff[11][5] = {
        {1.01108f, 75.2286f, -9.4671f, 0.000973445f, -1.01325f},
        {8.44447f, 0.0135054f, 1.11637f, 0.479139f, -9.30873f},
        {5.67124f, 0.0194668f, 1.12489f, 0.453551f, -6.53104f},
        {4.50086f, 0.0237758f, 1.09376f, 0.423665f, -5.36521f},
        {3.77781f, 0.0273286f, 1.0666f, 0.400267f, -4.65601f},
        {3.35338f, 0.0307392f, 0.989189f, 0.371074f, -4.24089f},
        {3.39605f, 0.0308828f, 0.845455f, 0.333813f, -4.28662f},
        {3.62583f, 0.0303531f, 0.79754f, 0.325588f, -4.51725f},
        {4.11524f, 0.0172797f, 0.701166f, 0.275242f, -5.04394f},
        {4.4012f, 0.0159469f, 0.685892f, 0.270221f, -5.32997f},
        {5.8912f, 0.0171461f, 0.66366f, 0.288735f, -6.78867f},
    };
    const float x = clamp(ndotv, 0.f, 1.f) * 20.f + 1.f; // fit domain [1, 21]
    const int ib = clamp(int(floor(roughness * 10.f)), 0, 10);
    const int it = min(ib + 1, 10);
    const float wb = float(it) - roughness * 10.f;
    const float tb = coeff[ib][4] + pow(x, coeff[ib][3]) +
                     coeff[ib][0] / (1.f + coeff[ib][1] * pow(x, coeff[ib][2]));
    const float tt = coeff[it][4] + pow(x, coeff[it][3]) +
                     coeff[it][0] / (1.f + coeff[it][1] * pow(x, coeff[it][2]));
    return 1.f - clamp(tb * wb + tt * (1.f - wb), 0.f, 1.f);
}

} // namespace merian
