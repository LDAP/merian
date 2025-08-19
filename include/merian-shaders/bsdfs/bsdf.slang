#pragma once

__exported import "merian-shaders/utils/random.slang";

namespace merian {

struct Albedo {
    float3 reflection;

    __init(const float3 reflected) {
        reflection = reflected;
    }
}

struct BSDFSample {
    float3 wo;
    float pdf;
}

static const float BSDF_MIN_COS_THETA = 1e-6f;

interface BSDF {

    /** Evaluates the BSDF. Assumes a local tangent space with n = (0, 0, 1), t = (1, 0, 0), b = (0,
     * 1, 0). The incident and outgoing direction point away from the shading location.
     *
     *  Returns in solid angle: f(wi, wo) * dot(wo, n)
     */
    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng);

    // -----------------------------

    /** Samples the BSDF. Assumes a local tangent space with n = (0, 0, 1), t = (1, 0, 0), b = (0,
     * 1, 0). The incident and outgoing direction point away from the shading location.
     *
     *  Returns in solid angle: f(wi, wo) * dot(wo, n)
     */
    Optional<BSDFSample> sample(const float3 wi, inout RandomGenerator rng);

    // -----------------------------

    /** Evaluates the pdf of the sample function of this BSDF. Assumes a local tangent space with n
     * = (0, 0, 1), t = (1, 0, 0), b = (0, 1, 0). The incident and outgoing direction point away
     * from the shading location.
     *
     *  Returns in solid angle.
     */
    float pdf(const float3 wi, const float3 wo);

    // -----------------------------

    Albedo get_albedo(const float3 wi);
}

}
