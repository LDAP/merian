import bsdf;

namespace merian {

struct MixBSDF<B0 : BSDF, B1 : BSDF> : BSDF {
    B0 bsdf0;
    B1 bsdf1;
    float weight;

    __init(B0 bsdf0, B1 bsdf1, float weight) {
        this.bsdf0 = bsdf0;
        this.bsdf1 = bsdf1;
        this.weight = weight;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        return lerp(bsdf0.eval(wi, wo, rng), bsdf1.eval(wi, wo, rng), weight);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        // weight > 0.0f check enables dead code elimination
        if (weight > 0.0f && rng.next_float() < weight) {
            return bsdf1.sample(wi, rng);
        }
        return bsdf0.sample(wi, rng);
    }

    float pdf(const float3 wi, const float3 wo) {
        return lerp(bsdf0.pdf(wi, wo), bsdf1.pdf(wi, wo), weight);
    }

    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        // weight > 0.0f check enables dead code elimination
        if (weight > 0.0f && rng.next_float() < weight) {
            if (let s = bsdf1.sample_eval(wi, rng)) {
                const float mix_pdf = lerp(bsdf0.pdf(wi, s.wo), s.pdf, weight);
                const float3 mix_eval = lerp(bsdf0.eval(wi, s.wo, rng), s.weight * s.pdf, weight);
                return ShadingFunctionSample(s.wo, mix_pdf, mix_eval / mix_pdf, s.lobe);
            }
            return none;
        }
        if (let s = bsdf0.sample_eval(wi, rng)) {
            const float mix_pdf = lerp(s.pdf, bsdf1.pdf(wi, s.wo), weight);
            const float3 mix_eval = lerp(s.weight * s.pdf, bsdf1.eval(wi, s.wo, rng), weight);
            return ShadingFunctionSample(s.wo, mix_pdf, mix_eval / mix_pdf, s.lobe);
        }
        return none;
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(
            lerp(bsdf0.get_albedo(wi).reflection, bsdf1.get_albedo(wi).reflection, weight));
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return lerp(bsdf0.get_preintegrated_specular(F0, wi),
                    bsdf1.get_preintegrated_specular(F0, wi), weight);
    }

    property float roughness {
        get { return min(bsdf0.roughness, bsdf1.roughness); }
    }
}

}
