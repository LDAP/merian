#pragma once

#include "merian-shaders/utils/math-constants.slangh"

import bsdf;
import brdf_lambert_diffuse;
import brdf_ggx;
import merian_shaders.utils.fresnel;

namespace merian {

// Lambert diffuse + GGX specular, Fresnel-weighted per the metallic-roughness model.
struct LambertGGXBRDF : BSDF {
    LambertDiffuseBRDF diffuse;
    GGXBRDF specular;

    __init(float3 diffuse_albedo, float3 F0, float alpha) {
        diffuse = LambertDiffuseBRDF(diffuse_albedo);
        specular = GGXBRDF(F0, alpha);
    }

    // Lobe-selection probability balanced by per-lobe luminance at wi.z, clamped for MIS safety.
    float lobe_spec_prob(const float wiz) {
        const float3 F = fresnel_schlick(wiz, specular.F0);
        const float F_lum = (F.r + F.g + F.b) * (1.0f / 3.0f);
        const float D_lum = ((1.0f - F.r) * diffuse.albedo.r
                           + (1.0f - F.g) * diffuse.albedo.g
                           + (1.0f - F.b) * diffuse.albedo.b) * (1.0f / 3.0f);
        return clamp(F_lum / max(F_lum + D_lum, 1e-6f), 0.1f, 0.9f);
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return float3(0.0f);
        }
        // (1 - F) splits energy between the GGX specular and Lambert diffuse lobes.
        const float3 h = normalize(wi + wo);
        const float3 F = fresnel_schlick(dot(wi, h), specular.F0);
        return specular.eval(wi, wo, rng) + (float3(1.0f) - F) * diffuse.eval(wi, wo, rng);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return none;
        }
        const float spec_prob = lobe_spec_prob(wi.z);
        if (rng.next_float() < spec_prob) {
            return specular.sample(wi, rng);
        }
        return diffuse.sample(wi, rng);
    }

    float pdf(const float3 wi, const float3 wo) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return 0.0f;
        }
        const float spec_prob = lobe_spec_prob(wi.z);
        return spec_prob * specular.pdf(wi, wo) + (1.0f - spec_prob) * diffuse.pdf(wi, wo);
    }

    Albedo get_albedo(const float3 wi) {
        const float3 F = fresnel_schlick(wi.z, specular.F0);
        return Albedo(F + (float3(1.0f) - F) * diffuse.albedo);
    }
}

}
