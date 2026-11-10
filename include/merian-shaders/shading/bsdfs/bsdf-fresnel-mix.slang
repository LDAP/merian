import merian_shaders.shading.bsdfs.bsdf;
import merian_shaders.utils.fresnel;
import merian_shaders.utils.math;

namespace merian {

struct FresnelMixBSDF<Layer : BSDF, Base : BSDF> : BSDF {
    Layer layer;
    Base base;
    float ior_f0;
    float weight;

    __init(Layer layer, Base base, float ior_f0, float weight = 1.0f) {
        this.layer = layer;
        this.base = base;
        this.ior_f0 = ior_f0;
        this.weight = weight;
    }

    float layer_share(const float cos_theta) {
        return weight * fresnel_schlick(cos_theta, ior_f0);
    }

    float3 eval_pdf(const float3 wi, const float3 wo, inout RandomGenerator rng, out float pdf) {
        BSDFLobe lobe;
        return eval_pdf_impl<false>(wi, wo, rng, pdf, lobe);
    }

    override float3 eval_pdf_lobe(const float3 wi,
                                  const float3 wo,
                                  inout RandomGenerator rng,
                                  out float pdf,
                                  out BSDFLobe lobe) {
        return eval_pdf_impl<true>(wi, wo, rng, pdf, lobe);
    }

    float3 eval_pdf_impl<let LOBE : bool>(const float3 wi,
                                          const float3 wo,
                                          inout RandomGenerator rng,
                                          out float pdf,
                                          out BSDFLobe lobe) {
        const float F_wi = layer_share(abs(wi.z));
        // a transmitted direction has no half vector with wi, so mix by the incidence angle
        const float F = wi.z * wo.z > 0.0f ? layer_share(dot(wi, normalize(wi + wo))) : F_wi;
        float base_pdf;
        float layer_pdf = 0.0f;
        BSDFLobe base_lobe;
        BSDFLobe layer_lobe = BSDFLobe::None;
        float3 layer_e = float3(0.0f);
        const float3 base_e = eval_pdf_part<Base, LOBE>(base, wi, wo, rng, base_pdf, base_lobe);
        if (weight > 0.0f) {
            layer_e = eval_pdf_part<Layer, LOBE>(layer, wi, wo, rng, layer_pdf, layer_lobe);
        }
        pdf = lerp(base_pdf, layer_pdf, F_wi);
        lobe = BSDFLobe::None;
        if (LOBE) {
            lobe = pick_lobe(rng, (1.0f - F_wi) * base_pdf, F_wi * layer_pdf, F_wi, base_lobe,
                             layer_lobe);
        }
        return lerp(base_e, layer_e, F);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        if (weight > 0.0f && rng.next_float() < layer_share(abs(wi.z))) {
            return layer.sample(wi, rng);
        }
        return base.sample(wi, rng);
    }

    float pdf(const float3 wi, const float3 wo) {
        const float layer_pdf = weight > 0.0f ? layer.pdf(wi, wo) : 0.0f;
        return lerp(base.pdf(wi, wo), layer_pdf, layer_share(abs(wi.z)));
    }

    override Optional<ShadingFunctionSample> sample_eval(const float3 wi,
                                                         inout RandomGenerator rng) {
        const float F = layer_share(abs(wi.z));
        const bool take_layer = weight > 0.0f && rng.next_float() < F;
        Optional<ShadingFunctionSample> sub =
            take_layer ? layer.sample_eval(wi, rng) : base.sample_eval(wi, rng);
        if (let s = sub) {
            // The Fresnel cancels against the probability of having picked that lobe.
            if (weight <= 0.0f || bsdf_lobe_is_delta(s.lobe)) {
                return ShadingFunctionSample(s.wo, (take_layer ? F : 1.0f - F) * s.pdf, s.weight,
                                             s.lobe);
            }
            float p;
            const float3 f = this.eval_pdf(wi, s.wo, rng, p);
            return ShadingFunctionSample(s.wo, p, f / p, s.lobe);
        }
        return none;
    }

    Albedo get_albedo(const float3 wi) {
        const float3 spec = weight * layer.get_preintegrated_specular(float3(ior_f0), wi);
        const Albedo transmitted = base.get_albedo(wi).scaled(1.0f - spec);
        return Albedo(transmitted.diffuse, transmitted.specular + spec);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return lerp(base.get_preintegrated_specular(F0, wi),
                    layer.get_preintegrated_specular(F0, wi), weight);
    }

    // Weighted RMS, by the share of the response the layer takes at normal incidence.
    property float effective_alpha {
        get {
            return sqrt(lerp(pow2(base.effective_alpha), pow2(layer.effective_alpha),
                             weight * ior_f0));
        }
    }

    property float specular_alpha {
        get {
            return sqrt(lerp(pow2(base.specular_alpha), pow2(layer.specular_alpha), weight));
        }
    }

    uint get_lobe_types() {
        uint lobes = base.get_lobe_types();
        if (weight > 0.0f) {
            lobes |= layer.get_lobe_types();
        }
        return lobes;
    }
}

} // namespace merian
