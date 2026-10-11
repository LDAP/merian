import merian_shaders.shading.bsdfs.bsdf;
import merian_shaders.utils.math;

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

    float3 eval_pdf(const float3 wi, const float3 wo, inout RandomGenerator rng, out float pdf) {
        float p0 = 0.0f;
        float p1 = 0.0f;
        float3 e0 = float3(0.0f);
        float3 e1 = float3(0.0f);
        if (weight < 1.0f) {
            e0 = bsdf0.eval_pdf(wi, wo, rng, p0);
        }
        if (weight > 0.0f) {
            e1 = bsdf1.eval_pdf(wi, wo, rng, p1);
        }
        pdf = lerp(p0, p1, weight);
        return lerp(e0, e1, weight);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        // weight > 0.0f check enables dead code elimination
        if (weight > 0.0f && rng.next_float() < weight) {
            return bsdf1.sample(wi, rng);
        }
        return bsdf0.sample(wi, rng);
    }

    float pdf(const float3 wi, const float3 wo) {
        const float p0 = weight < 1.0f ? bsdf0.pdf(wi, wo) : 0.0f;
        const float p1 = weight > 0.0f ? bsdf1.pdf(wi, wo) : 0.0f;
        return lerp(p0, p1, weight);
    }

    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        // weight > 0.0f check enables dead code elimination
        if (weight > 0.0f && rng.next_float() < weight) {
            if (let s = bsdf1.sample_eval(wi, rng)) {
                // The mix weight cancels against the probability of having picked that lobe.
                if (bsdf_lobe_is_delta(s.lobe)) {
                    return ShadingFunctionSample(s.wo, weight * s.pdf, s.weight, s.lobe);
                }
                float other_pdf = 0.0f;
                float3 other = float3(0.0f);
                if (weight < 1.0f) {
                    other = bsdf0.eval_pdf(wi, s.wo, rng, other_pdf);
                }
                const float mix_pdf = lerp(other_pdf, s.pdf, weight);
                const float3 mix_eval = lerp(other, s.weight * s.pdf, weight);
                return ShadingFunctionSample(s.wo, mix_pdf, mix_eval / mix_pdf, s.lobe);
            }
            return none;
        }
        if (let s = bsdf0.sample_eval(wi, rng)) {
            if (bsdf_lobe_is_delta(s.lobe)) {
                return ShadingFunctionSample(s.wo, (1.0f - weight) * s.pdf, s.weight, s.lobe);
            }
            float other_pdf = 0.0f;
            float3 other = float3(0.0f);
            if (weight > 0.0f) {
                other = bsdf1.eval_pdf(wi, s.wo, rng, other_pdf);
            }
            const float mix_pdf = lerp(s.pdf, other_pdf, weight);
            const float3 mix_eval = lerp(s.weight * s.pdf, other, weight);
            return ShadingFunctionSample(s.wo, mix_pdf, mix_eval / mix_pdf, s.lobe);
        }
        return none;
    }

    Albedo get_albedo(const float3 wi) {
        const Albedo a0 = bsdf0.get_albedo(wi);
        const Albedo a1 = bsdf1.get_albedo(wi);
        return Albedo(lerp(a0.diffuse, a1.diffuse, weight), lerp(a0.specular, a1.specular, weight));
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return lerp(bsdf0.get_preintegrated_specular(F0, wi),
                    bsdf1.get_preintegrated_specular(F0, wi), weight);
    }

    // Weighted RMS
    property float effective_alpha {
        get {
            return sqrt(lerp(pow2(bsdf0.effective_alpha), pow2(bsdf1.effective_alpha), weight));
        }
    }

    property float specular_alpha {
        get {
            return sqrt(lerp(pow2(bsdf0.specular_alpha), pow2(bsdf1.specular_alpha), weight));
        }
    }

    // A branch the weight never selects contributes none: ORing unconditionally would report
    // transmission on every opaque surface built by weighting a lobe to zero.
    uint get_lobe_types() {
        uint lobes = 0;
        if (weight < 1.0f) {
            lobes |= bsdf0.get_lobe_types();
        }
        if (weight > 0.0f) {
            lobes |= bsdf1.get_lobe_types();
        }
        return lobes;
    }
}

} // namespace merian
