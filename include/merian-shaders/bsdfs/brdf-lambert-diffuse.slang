#pragma once

#include "merian-shaders/utils/math-constants.slangh"

import "merian-shaders/bsdfs/bsdf.slang";
import "merian-shaders/utils/sampling.slang";

namespace merian {

// A lambert diffuse BRDF.
struct LambertDiffuseBRDF : BSDF {
    float3 albedo;

    __init(float3 albedo) {
        this.albedo = albedo;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return float3(0.0f);
        }

        return MERIAN_INV_PI * albedo * wo.z;
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        const float3 wo = sample_cos(rng.next_float2());
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return none;
        }
        return wo;
    }

    override Optional<BSDFSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        const float3 wo = sample_cos(rng.next_float2());
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return none;
        }

        return BSDFSample(wo, MERIAN_INV_PI * wo.z, albedo);
    }

    float pdf(const float3 wi, const float3 wo) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return 0.0f;
        }

        return MERIAN_INV_PI * wo.z;
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(albedo);
    }
}

}
