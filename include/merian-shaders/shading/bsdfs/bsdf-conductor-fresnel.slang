#pragma once

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

    Albedo get_albedo(const float3 wi) {
        const float3 F = fresnel_schlick(wi.z, f0);
        return Albedo(F * bsdf.get_albedo(wi).reflection);
    }
}

}
