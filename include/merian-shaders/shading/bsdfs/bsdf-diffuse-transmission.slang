#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.shading.bsdfs.bsdf;
import merian_shaders.colors.colorspaces;
import merian_shaders.utils.sampling;

namespace merian {

struct DiffuseTransmissionBSDF : BSDF {
    float3 reflectance;
    float3 transmittance;

    __init(const float3 reflectance, const float3 transmittance) {
        this.reflectance = reflectance;
        this.transmittance = transmittance;
    }

    float reflect_probability() {
        const float r = yuv_luminance(reflectance);
        const float t = yuv_luminance(transmittance);
        return r + t > 0.0f ? r / (r + t) : 1.0f;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (min(abs(wi.z), abs(wo.z)) < BSDF_MIN_COS_THETA) {
            return float3(0.0f);
        }
        return MERIAN_INV_PI * (wi.z * wo.z > 0.0f ? reflectance : transmittance) * abs(wo.z);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        if (abs(wi.z) < BSDF_MIN_COS_THETA) {
            return none;
        }
        float3 wo = sample_cos(rng);
        if (wo.z < BSDF_MIN_COS_THETA) {
            return none;
        }
        const float p = reflect_probability();
        const bool reflect = p >= 1.0f || (p > 0.0f && rng.next_float() < p);
        wo.z *= (wi.z > 0.0f) == reflect ? 1.0f : -1.0f;
        return wo;
    }

    float pdf(const float3 wi, const float3 wo) {
        if (min(abs(wi.z), abs(wo.z)) < BSDF_MIN_COS_THETA) {
            return 0.0f;
        }
        const float p = reflect_probability();
        return MERIAN_INV_PI * abs(wo.z) * (wi.z * wo.z > 0.0f ? p : 1.0f - p);
    }

    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        if (abs(wi.z) < BSDF_MIN_COS_THETA) {
            return none;
        }
        float3 wo = sample_cos(rng);
        if (wo.z < BSDF_MIN_COS_THETA) {
            return none;
        }
        const float p = reflect_probability();
        if (p >= 1.0f || (p > 0.0f && rng.next_float() < p)) {
            wo.z *= wi.z > 0.0f ? 1.0f : -1.0f;
            return ShadingFunctionSample(wo, MERIAN_INV_PI * abs(wo.z) * p, reflectance / p,
                                         BSDFLobe::DiffuseReflection);
        }
        wo.z *= wi.z > 0.0f ? -1.0f : 1.0f;
        return ShadingFunctionSample(wo, MERIAN_INV_PI * abs(wo.z) * (1.0f - p),
                                     transmittance / (1.0f - p), BSDFLobe::DiffuseTransmission);
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo::diffuse_only(reflectance + transmittance);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return float3(0.0f);
    }

    property float effective_alpha {
        get {
            return 1.0f;
        }
    }

    property float specular_alpha {
        get {
            return 1.0f;
        }
    }

    uint get_lobe_types() {
        uint lobes = 0;
        if (any(reflectance > 0.0f)) {
            lobes |= uint(BSDFLobe::DiffuseReflection);
        }
        if (any(transmittance > 0.0f)) {
            lobes |= uint(BSDFLobe::DiffuseTransmission);
        }
        return lobes;
    }
}

} // namespace merian
