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
    // the outgoing direction
    float3 wo;
    // probability density in solid angle
    float pdf;
    // sample weight, i.e. f(wi, wo) * dot(wo, n) / pdf = eval() / pdf().
    float3 weight;

    __init(const float3 wo, const float pdf, const float3 weight) {
        this.wo = wo;
        this.pdf = pdf;
        this.weight = weight;
    }
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
     * Returns an outgoing direction wo.
     */
    Optional<float3> sample(const float3 wi, inout RandomGenerator rng);

    /** Samples and evaluates the BSDF. Assumes a local tangent space with n = (0, 0, 1), t = (1, 0,
     * 0), b = (0, 1, 0). The incident and outgoing direction point away from the shading location.
     *
     */
    Optional<BSDFSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        Optional<float3> optional_wo = sample(wi, rng);

        if (let wo = optional_wo) {
            float p = pdf(wi, wo);
            return BSDFSample(wo, p, eval(wi, wo, rng) / p);
        }

        return none;
    }

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
