import bsdf;
import merian_shaders.utils.fresnel;
import merian_shaders.utils.math;

namespace merian {

// delta reflection / refraction
struct SmoothDielectricBSDF : BSDF {
    float eta; // relative index of refraction, incident / transmitted
    float3 transmission_color;

    __init(const float eta, const float3 transmission_color) {
        this.eta = eta;
        this.transmission_color = transmission_color;
    }

    float3 reflect_dir(const float3 wi) {
        return float3(-wi.x, -wi.y, wi.z);
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        return float3(0.0f);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return none;
        }
        if (rng.next_float() < fresnel_dielectric(wi.z, eta)) {
            return reflect_dir(wi);
        }
        const float3 wt = refract(-wi, float3(0.0f, 0.0f, 1.0f), eta);
        return all(wt == float3(0.0f)) ? reflect_dir(wi) : wt;
    }

    float pdf(const float3 wi, const float3 wo) {
        return 0.0f;
    }

    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return none;
        }
        if (rng.next_float() < fresnel_dielectric(wi.z, eta)) {
            return ShadingFunctionSample(reflect_dir(wi), 1.0f, float3(1.0f));
        }
        const float3 wt = refract(-wi, float3(0.0f, 0.0f, 1.0f), eta);
        if (all(wt == float3(0.0f))) {
            return ShadingFunctionSample(reflect_dir(wi), 1.0f,
                                         float3(1.0f)); // total internal reflection
        }
        return ShadingFunctionSample(wt, 1.0f, transmission_color);
    }

    Albedo get_albedo(const float3 wi) {
        // Directional albedo of the delta interface: reflected fraction + transmitted rest, the
        // latter tinted and scaled by the eta^2 radiance factor.
        const float3 f0 = float3(pow2((1.0f - eta) / (1.0f + eta)));
        const float3 spec = fresnel_schlick(wi.z, f0);
        return Albedo(spec + (1.0f - spec) * pow2(eta) * transmission_color);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return fresnel_schlick(wi.z, F0);
    }
}

} // namespace merian
