#pragma once

import bsdf;
import merian_shaders.utils.fresnel;

namespace merian {

struct FresnelMixBSDF<Layer : BSDF, Base : BSDF> : BSDF {
    Layer layer;
    Base base;
    float ior_f0;

    __init(Layer layer, Base base, float ior_f0) {
        this.layer = layer;
        this.base = base;
        this.ior_f0 = ior_f0;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        const float3 h = normalize(wi + wo);
        const float F = fresnel_schlick(dot(wi, h), ior_f0);
        return lerp(base.eval(wi, wo, rng), layer.eval(wi, wo, rng), F);
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        const float F = fresnel_schlick(wi.z, ior_f0);
        if (rng.next_float() < F) {
            return layer.sample(wi, rng);
        }
        return base.sample(wi, rng);
    }

    float pdf(const float3 wi, const float3 wo) {
        const float F = fresnel_schlick(wi.z, ior_f0);
        return lerp(base.pdf(wi, wo), layer.pdf(wi, wo), F);
    }

    Albedo get_albedo(const float3 wi) {
        const float F = fresnel_schlick(wi.z, ior_f0);
        return Albedo(lerp(base.get_albedo(wi).reflection, layer.get_albedo(wi).reflection, F));
    }
}

}
