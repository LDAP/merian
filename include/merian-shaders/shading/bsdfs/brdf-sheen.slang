#include "merian-shaders/utils/math-constants.slangh"

import bsdf;
import merian_shaders.utils.sampling;

namespace merian {

// Microfacet sheen BRDF (Estevez & Kulla 2017).
// TODO: replace cosine sampling with LTC sheen (Zeltner et al. 2022) for lower variance and
// energy-correct layering.
struct SheenBRDF : BSDF {
    float3 color;
    float roughness;

    __init(float3 color, float roughness) {
        this.color = color;
        this.roughness = max(roughness, 0.07f);
    }

    float sheen_D(const float ndoth) {
        const float inv_alpha = 1.f / roughness;
        const float cos2h = ndoth * ndoth;
        const float sin2h = max(1.f - cos2h, 0.0078125f);
        return (2.f + inv_alpha) * pow(sin2h, inv_alpha * 0.5f) * (0.5f * MERIAN_INV_PI);
    }

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
        return 1.f / (1.f + pow(sheen_lambda(ndoto), 1.f + 2.f * pow(1.f - ndoto, 8.f)) +
                      sheen_lambda(ndoti));
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return float3(0.f);
        }
        const float3 h = normalize(wi + wo);
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
        return Albedo(color);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return float3(0.f);
    }
}

}
