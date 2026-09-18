import bsdf;
import brdf_sheen;

namespace merian {

// Layers a sheen lobe over an arbitrary base BSDF with energy-conserving albedo scaling:
//
//     scaling = min(1 - max(color) * E(ndotv), 1 - max(color) * E(ndotl))
//     f       = f_sheen + scaling * f_base
//
// E is the sheen directional albedo; the scaling darkens the base so the layer conserves energy.
// With a zero sheen colour the lobe and scaling fold away and this reduces to the base.
struct SheenLayerBSDF<Base : BSDF> : BSDF {
    SheenBRDF sheen;
    Base base;
    float sheen_alpha;
    float max_sheen_color; // max channel of the sheen colour

    __init(SheenBRDF sheen, Base base, float sheen_alpha, float max_sheen_color) {
        this.sheen = sheen;
        this.base = base;
        this.sheen_alpha = sheen_alpha;
        this.max_sheen_color = max_sheen_color;
    }

    // Energy-conserving darkening of the base, evaluated for a single direction cosine.
    float albedo_scaling(const float ndot) {
        return 1.f - max_sheen_color * sheen_directional_albedo(ndot, sheen_alpha);
    }

    // Probability of sampling the sheen lobe (rest goes to the base). Approximates the sheen's
    // share of the reflected energy; 0 when there is no sheen, so the sheen path is eliminated.
    float prob_sheen(const float ndotv) {
        return clamp(max_sheen_color * sheen_directional_albedo(ndotv, sheen_alpha), 0.f, 0.9f);
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        const float scaling = min(albedo_scaling(wi.z), albedo_scaling(wo.z));
        return sheen.eval(wi, wo, rng) + base.eval(wi, wo, rng) * scaling;
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        const float p = prob_sheen(wi.z);
        // p > 0.0f check enables dead code elimination when sheen is disabled.
        if (p > 0.0f && rng.next_float() < p) {
            return sheen.sample(wi, rng);
        }
        return base.sample(wi, rng);
    }

    float pdf(const float3 wi, const float3 wo) {
        return lerp(base.pdf(wi, wo), sheen.pdf(wi, wo), prob_sheen(wi.z));
    }

    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        // Sample sheen or base, but evaluate the combined pdf/weight. Delegate to the chosen
        // sub-BSDF so its lobe classification propagates.
        const float p = prob_sheen(wi.z);
        const bool take_sheen = p > 0.0f && rng.next_float() < p;
        Optional<ShadingFunctionSample> sub =
            take_sheen ? sheen.sample_eval(wi, rng) : base.sample_eval(wi, rng);
        if (let s = sub) {
            // Only the scaled base reaches a delta direction, and its scaling does not cancel.
            if (bsdf_lobe_is_delta(s.lobe)) {
                const float scaling = min(albedo_scaling(wi.z), albedo_scaling(s.wo.z));
                return ShadingFunctionSample(s.wo, (1.0f - p) * s.pdf,
                                             scaling * s.weight / (1.0f - p), s.lobe);
            }
            const float pdf = this.pdf(wi, s.wo);
            return ShadingFunctionSample(s.wo, pdf, this.eval(wi, s.wo, rng) / pdf, s.lobe);
        }
        return none;
    }

    Albedo get_albedo(const float3 wi) {
        const Albedo scaled_base = base.get_albedo(wi).scaled(float3(albedo_scaling(wi.z)));
        const Albedo sheen_albedo = sheen.get_albedo(wi);
        return Albedo(sheen_albedo.diffuse + scaled_base.diffuse,
                      sheen_albedo.specular + scaled_base.specular);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return base.get_preintegrated_specular(F0, wi);
    }

    property float effective_alpha {
        get {
            return base.effective_alpha;
        }
    }

    property float specular_alpha {
        get {
            return base.specular_alpha;
        }
    }

    uint get_lobe_types() {
        return uint(BSDFLobe::DiffuseReflection) | base.get_lobe_types();
    }
}

} // namespace merian
