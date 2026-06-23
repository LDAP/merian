#include "merian-shaders/utils/math-constants.slangh"

import bsdf;
import merian_shaders.shading.bsdfs.brdf_ggx;
import merian_shaders.utils.fresnel;
import merian_shaders.utils.math;

namespace merian {

// Conductor specular layer; its Fresnel blends Schlick with thin-film reflectance by `iridescence`.
struct IridescentConductorBSDF<Inner : BSDF> : BSDF {
    Inner bsdf;
    float3 f0;
    float film_ior;
    float thickness;
    float iridescence;

    __init(Inner bsdf, float3 f0, float film_ior, float thickness, float iridescence) {
        this.bsdf = bsdf;
        this.f0 = f0;
        this.film_ior = film_ior;
        this.thickness = thickness;
        this.iridescence = iridescence;
    }

    // Both Schlick and the thin-film reflectance at the microfacet angle (VdotH); the thin-film
    // replaces the microfacet Fresnel term, per Belcour & Barla's microfacet model.
    float3 fresnel(const float vdoth) {
        const float3 schlick = fresnel_schlick(vdoth, f0);
        // No film: plain Fresnel; also folds the thin-film term away when iridescence is constant
        // 0.
        if (iridescence <= 0.0f) {
            return schlick;
        }
        return lerp(schlick, fresnel_iridescent(1.0f, film_ior, f0, thickness, vdoth), iridescence);
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        const float3 h = normalize(wi + wo);
        return fresnel(dot(wi, h)) * bsdf.eval(wi, wo, rng);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        return bsdf.sample(wi, rng);
    }

    float pdf(const float3 wi, const float3 wo) {
        return bsdf.pdf(wi, wo);
    }

    override Optional<ShadingFunctionSample> sample_eval(const float3 wi,
                                                         inout RandomGenerator rng) {
        if (let s = bsdf.sample_eval(wi, rng)) {
            ShadingFunctionSample r = s;
            const float3 h = normalize(wi + r.wo);
            r.weight *= fresnel(dot(wi, h));
            return r;
        }
        return none;
    }

    Albedo get_albedo(const float3 wi) {
        const float3 spec = bsdf.get_preintegrated_specular(f0, wi);
        if (iridescence <= 0.0f) {
            return Albedo(spec);
        }
        return Albedo(spec * fresnel(wi.z) / max(fresnel_schlick(wi.z, f0), 1e-4f));
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return bsdf.get_preintegrated_specular(F0, wi);
    }

    property float roughness {
        get { return bsdf.roughness; }
    }
}

// Specular reflection layered over a diffuse/transmission base. The reflect/transmit split uses the
// full (TIR-aware) dielectric Fresnel; `iridescence` blends the split towards the thin-film
// reflectance via the glTF rgb_mix. Evaluated at the microfacet angle (VdotH).
struct IridescentFresnelMixBSDF<Layer : BSDF, Base : BSDF> : BSDF {
    Layer layer; // specular (top)
    Base base;   // diffuse / transmission (bottom)
    float ior_f0;
    float eta; // relative index of refraction (incident / transmitted)
    float film_ior;
    float thickness;
    float iridescence;

    __init(Layer layer,
           Base base,
           float ior_f0,
           float eta,
           float film_ior,
           float thickness,
           float iridescence) {
        this.layer = layer;
        this.base = base;
        this.ior_f0 = ior_f0;
        this.eta = eta;
        this.film_ior = film_ior;
        this.thickness = thickness;
        this.iridescence = iridescence;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        const float3 base_e = base.eval(wi, wo, rng);
        const float3 layer_e = layer.eval(wi, wo, rng);

        // Reflect/transmit split at the microfacet angle (VdotH); the full dielectric form keeps
        // TIR -> 1 so transmissive glass reflects past the critical angle instead of going dark.
        const float widoth = abs(dot(wi, normalize(wi + wo)));
        const float f = fresnel_dielectric(widoth, eta);
        const float3 dielectric = lerp(base_e, layer_e, f);

        // No film: plain Fresnel mix; also folds the thin-film term away when iridescence is
        // constant 0.
        if (iridescence <= 0.0f) {
            return dielectric;
        }

        // Per-channel energy split: the reflected fraction is irf, the diffuse base carries the
        // complementary 1 - irf (Belcour); the transmitted thin-film colour is the complement.
        const float3 irf = fresnel_iridescent(1.0f, film_ior, float3(ior_f0), thickness, widoth);
        const float3 iridescent = (1.0f - irf) * base_e + irf * layer_e;

        return lerp(dielectric, iridescent, iridescence);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        // Macro-normal Fresnel as the lobe-selection probability (no half vector exists yet);
        // pdf() mirrors it, the sample weight corrects the difference to eval.
        const float f = fresnel_dielectric(wi.z, eta);
        if (rng.next_float() < f) {
            return layer.sample(wi, rng);
        }
        return base.sample(wi, rng);
    }

    float pdf(const float3 wi, const float3 wo) {
        const float f = fresnel_dielectric(wi.z, eta);
        return lerp(base.pdf(wi, wo), layer.pdf(wi, wo), f);
    }

    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        // Sample layer or base by Fresnel, evaluate the combined pdf/weight, propagate the
        // chosen sub-BSDF's lobe.
        const float f = fresnel_dielectric(wi.z, eta);
        Optional<ShadingFunctionSample> sub =
            rng.next_float() < f ? layer.sample_eval(wi, rng) : base.sample_eval(wi, rng);
        if (let s = sub) {
            const float pdf = this.pdf(wi, s.wo);
            return ShadingFunctionSample(s.wo, pdf, this.eval(wi, s.wo, rng) / pdf, s.lobe);
        }
        return none;
    }

    Albedo get_albedo(const float3 wi) {
        float3 spec = layer.get_preintegrated_specular(float3(ior_f0), wi);
        if (iridescence > 0.0f) {
            // Modulate the specular fraction by the thin-film hue at the view angle.
            const float3 irf = fresnel_iridescent(1.0f, film_ior, float3(ior_f0), thickness, wi.z);
            const float die = fresnel_dielectric(wi.z, eta);
            spec = clamp(spec * lerp(float3(1.0f), irf / max(die, 1e-4f), iridescence), 0.0f, 1.0f);
        }
        return Albedo(spec + (1.0f - spec) * base.get_albedo(wi).reflection);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return layer.get_preintegrated_specular(F0, wi);
    }

    property float roughness {
        get { return min(layer.roughness, base.roughness); }
    }
}

// Coupled rough dielectric (GGX glass) whose interface reflectance is the Belcour-Barla thin-film
// reflectance (RGB, at the microfacet angle). The microfacet is sampled first, so a facet in total
// internal reflection reflects instead of terminating the path. Energy conserves per channel:
// reflection carries R, transmission carries 1 - R. eta = incident / transmitted index.
struct IridescentRoughDielectricBSDF : BSDF {
    float alpha;
    float eta;
    float3 tint; // transmission colour
    float3 base_f0;
    float film_ior;
    float thickness;
    float iridescence;

    __init(const float alpha,
           const float eta,
           const float3 tint,
           const float3 base_f0,
           const float film_ior,
           const float thickness,
           const float iridescence) {
        this.alpha = alpha;
        this.eta = eta;
        this.tint = tint;
        this.base_f0 = base_f0;
        this.film_ior = film_ior;
        this.thickness = thickness;
        this.iridescence = iridescence;
    }

    // Interface reflectance at the microfacet angle: plain dielectric Fresnel blended towards the
    // thin-film reflectance by `iridescence`.
    float3 reflectance(const float widoth) {
        const float dielectric = fresnel_dielectric(widoth, eta);
        if (iridescence <= 0.0f) {
            return float3(dielectric);
        }
        return lerp(float3(dielectric),
                    fresnel_iridescent(1.0f, film_ior, base_f0, thickness, widoth), iridescence);
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return float3(0.0f);
        }
        if (wo.z > BSDF_MIN_COS_THETA) {
            // Reflection: thin-film reflectance at the reflection half-vector weights the GGX lobe.
            const float3 h = normalize(wi + wo);
            return reflectance(abs(dot(wi, h))) * GGXBRDF(alpha).eval(wi, wo, rng);
        }
        if (wo.z < -BSDF_MIN_COS_THETA) {
            // Refraction: (1 - R) per channel times the GGX transmission lobe (with the eta^2
            // term).
            float3 h = normalize(wo + wi * eta);
            h *= sign(h.z);
            return (1.0f - reflectance(abs(dot(wi, h)))) *
                   GGXBTDF(alpha, eta, tint).eval(wi, wo, rng);
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
            const float p = max3(reflectance(abs(dot(wi, h))));
            return p * ggx_VNDF_pdf_wo(wi.z, pow2(max(0.0f, h.z)), alpha_squared);
        }
        if (wo.z < -BSDF_MIN_COS_THETA) {
            float3 h = normalize(wo + wi * eta);
            h *= sign(h.z);
            const float p = max3(reflectance(abs(dot(wi, h))));
            const float wodoth = dot(wo, h);
            const float denom = wodoth + eta * dot(wi, h);
            return (1.0f - p) * ggx_VNDF_H_pdf(wi, h, alpha_squared) * abs(wodoth) / pow2(denom);
        }
        return 0.0f;
    }

    override Optional<ShadingFunctionSample> sample_eval(const float3 wi,
                                                         inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return none;
        }
        const float alpha_squared = pow2(alpha);
        const float3 wm = ggx_VNDF_sample_H(wi, alpha, rng.next_float2());
        const float widoth = dot(wi, wm);
        const float3 R = reflectance(abs(widoth));
        const float p = max3(R); // scalar selection probability; full TIR -> 1 -> always reflect

        if (rng.next_float() < p) {
            // Reflection. Weight = (R / p) tint times the bare VNDF weight G2 / G1.
            const float3 wo = reflect2(wi, wm, widoth);
            if (wo.z < BSDF_MIN_COS_THETA) {
                return none;
            }
            const float g = smith_g2_over_widotn(wo.z, wi.z, alpha_squared) /
                            smith_g1_over_widotn(wi.z, alpha_squared);
            const float pdf = p * ggx_VNDF_pdf_wo(wi.z, pow2(wm.z), alpha_squared);
            return ShadingFunctionSample(wo, pdf, (R / p) * g, BSDFLobe::Glossy);
        }

        // Refraction. Real TIR was already routed to reflection (p == 1); a refraction failure here
        // is a numerical edge near the critical angle and contributes negligibly.
        const float3 wo = refract(-wi, wm, eta);
        if (all(wo == float3(0.0f)) || wo.z > -BSDF_MIN_COS_THETA) {
            return none;
        }
        const float wodoth = dot(wo, wm);
        const float denom = wodoth + eta * widoth;
        const float pdf =
            (1.0f - p) * ggx_VNDF_H_pdf(wi, wm, alpha_squared) * abs(wodoth) / pow2(denom);
        // Weight = ((1 - R) / (1 - p)) per channel times the eta^2 radiance factor and tint.
        const float g = smith_g2_over_widotn(abs(wo.z), wi.z, alpha_squared) /
                        smith_g1_over_widotn(wi.z, alpha_squared);
        return ShadingFunctionSample(wo, pdf, ((1.0f - R) / (1.0f - p)) * pow2(eta) * tint * g,
                                     BSDFLobe::Glossy);
    }

    Albedo get_albedo(const float3 wi) {
        // Reflected fraction + transmitted rest; the latter carries the eta^2 radiance factor and
        // tint. The thin-film hue (at the view angle) modulates the reflected fraction.
        const float3 f0 = float3(pow2((1.0f - eta) / (1.0f + eta)));
        float3 spec = GGXBRDF(alpha).get_preintegrated_specular(f0, wi);
        if (iridescence > 0.0f) {
            spec = clamp(spec * reflectance(wi.z) / max(fresnel_dielectric(wi.z, eta), 1e-4f), 0.0f,
                         1.0f);
        }
        return Albedo(spec + (1.0f - spec) * pow2(eta) * tint);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return GGXBRDF(alpha).get_preintegrated_specular(F0, wi);
    }

    property float roughness {
        get { return sqrt(alpha); }
    }
}

} // namespace merian
