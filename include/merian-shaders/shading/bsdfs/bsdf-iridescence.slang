import bsdf;
import merian_shaders.utils.fresnel;

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

    float3 fresnel(const float cos_theta) {
        const float3 schlick = fresnel_schlick(cos_theta, f0);
        // No film: plain Fresnel; also folds the thin-film term away when iridescence is constant 0.
        if (iridescence <= 0.0f) {
            return schlick;
        }
        return lerp(schlick, fresnel_iridescent(1.0f, film_ior, f0, thickness, cos_theta),
                    iridescence);
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

    override Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        if (let s = bsdf.sample_eval(wi, rng)) {
            ShadingFunctionSample r = s;
            const float3 h = normalize(wi + r.wo);
            r.weight *= fresnel(dot(wi, h));
            return r;
        }
        return none;
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(bsdf.get_preintegrated_specular(f0, wi));
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return bsdf.get_preintegrated_specular(F0, wi);
    }
}

// Dielectric specular over a diffuse/transmission base. The split uses the full (TIR-aware)
// dielectric Fresnel; `iridescence` blends in the thin-film reflectance.
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

        // Dielectric reflectance at the macro normal (consistent with sample/pdf; TIR -> 1).
        const float f = fresnel_dielectric(wi.z, eta);
        const float3 dielectric = lerp(base_e, layer_e, f);

        // No film: plain Fresnel mix; also folds the thin-film term away when iridescence is constant 0.
        if (iridescence <= 0.0f) {
            return dielectric;
        }

        const float3 irf = fresnel_iridescent(1.0f, film_ior, float3(ior_f0), thickness, wi.z);
        const float irf_max = max(irf.x, max(irf.y, irf.z));
        const float3 iridescent = (1.0f - irf_max) * base_e + irf * layer_e;

        return lerp(dielectric, iridescent, iridescence);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
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
        return sample_eval_impl<This>(this, wi, rng);
    }

    Albedo get_albedo(const float3 wi) {
        const float3 spec = layer.get_preintegrated_specular(float3(ior_f0), wi);
        return Albedo(spec + (1.0f - spec) * base.get_albedo(wi).reflection);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return layer.get_preintegrated_specular(F0, wi);
    }
}

}
