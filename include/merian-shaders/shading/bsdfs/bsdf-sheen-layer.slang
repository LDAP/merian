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
    float sheen_roughness; // perceptual, for the directional-albedo lookup
    float max_sheen_color; // max channel of the sheen colour

    __init(SheenBRDF sheen, Base base, float sheen_roughness, float max_sheen_color) {
        this.sheen = sheen;
        this.base = base;
        this.sheen_roughness = sheen_roughness;
        this.max_sheen_color = max_sheen_color;
    }

    // Energy-conserving darkening of the base, evaluated for a single direction cosine.
    float albedo_scaling(const float ndot) {
        return 1.f - max_sheen_color * sheen_directional_albedo(ndot, sheen_roughness);
    }

    // Probability of sampling the sheen lobe (rest goes to the base). Approximates the sheen's
    // share of the reflected energy; 0 when there is no sheen, so the sheen path is eliminated.
    float prob_sheen(const float ndotv) {
        return clamp(max_sheen_color * sheen_directional_albedo(ndotv, sheen_roughness), 0.f, 0.9f);
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
        return sample_eval_impl<This>(this, wi, rng);
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(sheen.get_albedo(wi).reflection +
                      albedo_scaling(wi.z) * base.get_albedo(wi).reflection);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return base.get_preintegrated_specular(F0, wi);
    }
}

}
