import merian_shaders.shading.bsdfs.bsdf;
import merian_shaders.utils.fresnel;
import merian_shaders.utils.math;

namespace merian {

// Dielectric of negligible thickness.
struct ThinDielectricBSDF : BSDF {
    float eta; // relative index of refraction, surrounding / dielectric
    float3 transmission_color;

    __init(const float eta, const float3 transmission_color) {
        this.eta = eta;
        this.transmission_color = transmission_color;
    }

    float reflectance(const float cos_theta) {
        const float R = fresnel_dielectric(abs(cos_theta), eta);
        return R < 1.0f ? R + pow2(1.0f - R) * R / (1.0f - pow2(R)) : 1.0f;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        return float3(0.0f);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        if (abs(wi.z) < BSDF_MIN_COS_THETA) {
            return none;
        }
        return rng.next_float() < reflectance(wi.z) ? reflect_z(wi) : -wi;
    }

    float pdf(const float3 wi, const float3 wo) {
        return 0.0f;
    }

    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        if (abs(wi.z) < BSDF_MIN_COS_THETA) {
            return none;
        }
        const float R = reflectance(wi.z);
        if (rng.next_float() < R) {
            return ShadingFunctionSample(reflect_z(wi), R, float3(1.0f), BSDFLobe::DeltaReflection);
        }
        return ShadingFunctionSample(-wi, 1.0f - R, transmission_color,
                                     BSDFLobe::DeltaTransmission);
    }

    Albedo get_albedo(const float3 wi) {
        const float R = reflectance(wi.z);
        return Albedo::specular_only(R + (1.0f - R) * transmission_color);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return fresnel_schlick(abs(wi.z), F0);
    }

    property float effective_alpha {
        get {
            return 0.0f;
        }
    }

    property float specular_alpha {
        get {
            return 0.0f;
        }
    }

    uint get_lobe_types() {
        return uint(BSDFLobe::DeltaReflection) | uint(BSDFLobe::DeltaTransmission);
    }
}

} // namespace merian
