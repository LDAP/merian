import bsdf;

namespace merian {

// Scales a BSDF by a constant color.
struct TintBSDF<Base : BSDF> : BSDF {
    Base base;
    float3 tint;

    __init(Base base, float3 tint) {
        this.base = base;
        this.tint = tint;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        return tint * base.eval(wi, wo, rng);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        return base.sample(wi, rng);
    }

    float pdf(const float3 wi, const float3 wo) {
        return base.pdf(wi, wo);
    }

    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        if (let s = base.sample_eval(wi, rng)) {
            return ShadingFunctionSample(s.wo, s.pdf, tint * s.weight, s.lobe);
        }
        return none;
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(tint * base.get_albedo(wi).reflection);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return tint * base.get_preintegrated_specular(F0, wi);
    }

    property float roughness {
        get {
            return base.roughness;
        }
    }
}

} // namespace merian
