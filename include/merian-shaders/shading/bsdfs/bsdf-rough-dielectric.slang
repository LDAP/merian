#include "merian-shaders/utils/math-constants.slangh"

import bsdf;
import merian_shaders.shading.bsdfs.brdf_ggx;
import merian_shaders.utils.fresnel;
import merian_shaders.utils.math;

namespace merian {

// Rough dielectric (GGX glass: coupled reflection + refraction). The microfacet is sampled first
// and the Fresnel evaluated at it, so a microfacet in total internal reflection reflects instead of
// terminating the path. eta = incident / transmitted index.
struct RoughDielectricBSDF : BSDF {
    float alpha;
    float eta;
    float3 tint; // transmission colour

    __init(const float alpha, const float eta, const float3 tint) {
        this.alpha = alpha;
        this.eta = eta;
        this.tint = tint;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return float3(0.0f);
        }
        if (wo.z > BSDF_MIN_COS_THETA) {
            // Reflection: Fresnel at the reflection half-vector, weighting the GGX reflection lobe.
            const float3 h = normalize(wi + wo);
            const float R = fresnel_dielectric(abs(dot(wi, h)), eta);
            return R * GGXBRDF(alpha).eval(wi, wo, rng);
        }
        if (wo.z < -BSDF_MIN_COS_THETA) {
            // Refraction: 1 - R times the GGX transmission lobe (which carries the eta^2 factor).
            float3 h = normalize(wo + wi * eta);
            h *= sign(h.z);
            const float R = fresnel_dielectric(abs(dot(wi, h)), eta);
            return (1.0f - R) * GGXBTDF(alpha, eta, tint).eval(wi, wo, rng);
        }
        return float3(0.0f);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        if (let s = sample_eval(wi, rng)) {
            return s.wo;
        }
        return none;
    }

    float pdf(const float3 wi, const float3 wo) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return 0.0f;
        }
        const float alpha_squared = pow2(alpha);
        if (wo.z > BSDF_MIN_COS_THETA) {
            const float3 h = normalize(wi + wo);
            const float R = fresnel_dielectric(abs(dot(wi, h)), eta);
            return R * ggx_VNDF_pdf_wo(wi.z, pow2(max(0.0f, h.z)), alpha_squared);
        }
        if (wo.z < -BSDF_MIN_COS_THETA) {
            float3 h = normalize(wo + wi * eta);
            h *= sign(h.z);
            const float R = fresnel_dielectric(abs(dot(wi, h)), eta);
            const float wodoth = dot(wo, h);
            const float denom = wodoth + eta * dot(wi, h);
            return (1.0f - R) * ggx_VNDF_H_pdf(wi, h, alpha_squared) * abs(wodoth) / pow2(denom);
        }
        return 0.0f;
    }

    override Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return none;
        }
        const float alpha_squared = pow2(alpha);
        const float3 wm = ggx_VNDF_sample_H(wi, alpha, rng.next_float2());
        const float widoth = dot(wi, wm);
        const float R = fresnel_dielectric(abs(widoth), eta);

        if (rng.next_float() < R) {
            // Reflection. The Fresnel R cancels in the weight (we sample with probability R), so
            // the weight is the bare VNDF weight G2 / G1.
            const float3 wo = reflect2(wi, wm, widoth);
            if (wo.z < BSDF_MIN_COS_THETA) {
                return none;
            }
            const float weight = smith_g2_over_widotn(wo.z, wi.z, alpha_squared) /
                                 smith_g1_over_widotn(wi.z, alpha_squared);
            const float pdf = R * ggx_VNDF_pdf_wo(wi.z, pow2(wm.z), alpha_squared);
            return ShadingFunctionSample(wo, pdf, float3(weight));
        }

        // Refraction. Real TIR was already routed to reflection (R == 1); a refraction failure here
        // is a numerical edge near the critical angle and contributes negligibly.
        const float3 wo = refract(-wi, wm, eta);
        if (all(wo == float3(0.0f)) || wo.z > -BSDF_MIN_COS_THETA) {
            return none;
        }
        const float wodoth = dot(wo, wm);
        const float denom = wodoth + eta * widoth;
        const float pdf =
            (1.0f - R) * ggx_VNDF_H_pdf(wi, wm, alpha_squared) * abs(wodoth) / pow2(denom);
        // (1 - R) cancels; the eta^2 radiance factor stays (same as GGXBTDF).
        const float weight = pow2(eta) * smith_g2_over_widotn(abs(wo.z), wi.z, alpha_squared) /
                             smith_g1_over_widotn(wi.z, alpha_squared);
        return ShadingFunctionSample(wo, pdf, tint * weight);
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(tint);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return float3(0.0f);
    }
}

}
