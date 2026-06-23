import bsdf;
import merian_shaders.utils.fresnel;

namespace merian {

struct ConductorFresnelBSDF<Inner : BSDF> : BSDF {
    Inner bsdf;
    float3 f0;

    __init(Inner bsdf, float3 f0) {
        this.bsdf = bsdf;
        this.f0 = f0;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        const float3 h = normalize(wi + wo);
        const float3 F = fresnel_schlick(dot(wi, h), f0);
        return F * bsdf.eval(wi, wo, rng);
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
            r.weight *= fresnel_schlick(dot(wi, h), f0);
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

    property float roughness {
        get { return bsdf.roughness; }
    }
}

}
