#include "merian-shaders/utils/math-constants.slangh"

import bsdf;
import merian_shaders.utils.sampling;

namespace merian {

// Microfacet sheen BRDF (Estevez & Kulla 2017).
struct SheenBRDF : BSDF {
    float3 color;
    float alpha; // squared roughness, clamped away from zero

    __init(float3 color, float roughness) {
        this.color = color;
        this.alpha = max(roughness * roughness, 1e-6f);
    }

    float sheen_D(const float ndoth) {
        const float inv_alpha = 1.f / alpha;
        const float cos2h = ndoth * ndoth;
        const float sin2h = max(1.f - cos2h, 0.0078125f);
        return (2.f + inv_alpha) * pow(sin2h, inv_alpha * 0.5f) * (0.5f * MERIAN_INV_PI);
    }

    // Rational fit for the visibility lambda term, parameterised by (1 - alpha)^2.
    float sheen_L(const float x) {
        const float r = (1.f - alpha) * (1.f - alpha);
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
        return 1.f / (1.f + sheen_lambda(ndoto) + sheen_lambda(ndoti));
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
        return sample_eval_impl<This>(this, wi, rng);
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(color * sheen_directional_albedo(wi.z, sqrt(alpha)));
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return float3(0.f);
    }
}

// Analytic fit of the sheen directional albedo (hemispherical reflectance).
float sheen_directional_albedo(const float ndotv, const float roughness) {
    const float cosv = clamp(ndotv, 0.f, 1.f);
    const float r2 = roughness * roughness;
    const float a = roughness < 0.25f ? -339.2f * r2 + 161.4f * roughness - 25.9f
                                      : -8.48f * r2 + 14.3f * roughness - 9.95f;
    const float b = roughness < 0.25f ? 44.0f * r2 - 23.7f * roughness + 3.26f
                                      : 1.97f * r2 - 3.27f * roughness + 0.72f;
    const float dg = exp(a * cosv + b) + (roughness < 0.25f ? 0.0f : 0.1f * (roughness - 0.25f));
    return clamp(dg * MERIAN_INV_PI, 0.f, 1.f);
}

}
