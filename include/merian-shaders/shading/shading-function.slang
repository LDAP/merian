#pragma once

__exported import merian_shaders.utils.random;

namespace merian {

// A sample from a BSDF or PhaseFunction.
struct ShadingFunctionSample {
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

// Common base for BSDFs and PhaseFunctions.
interface ShadingFunction {

    /** Evaluates the ShadingFunction. BSDFs assume a local tangent space with n = (0, 0, 1), t =
     * (1, 0, 0), b = (0, 1, 0). The incident and outgoing direction point away from the shading
     * location.
     *
     *  Returns solid angle: f(wi, wo) * dot(wo, n)
     */
    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng);

    // -----------------------------

    /** Samples the ShadingFunction. BSDFs assume a local tangent space with n = (0, 0, 1), t = (1,
     * 0, 0), b = (0, 1, 0). The incident and outgoing direction point away from the shading
     * location.
     *
     * Returns an outgoing direction wo.
     */
    Optional<float3> sample(const float3 wi, inout RandomGenerator rng);

    // -----------------------------

    /** Evaluates the pdf of the ShadingFunction. BSDFs assume a local tangent space with n
     * = (0, 0, 1), t = (1, 0, 0), b = (0, 1, 0). The incident and outgoing direction point away
     * from the shading location.
     *
     *  Returns in solid angle.
     */
    float pdf(const float3 wi, const float3 wo);

    /** Samples and evaluates the ShadingFunction. BSDFs assume a local tangent space with n = (0,
     * 0, 1), t = (1, 0, 0), b = (0, 1, 0). The incident and outgoing direction point away from the
     * shading location.
     *
     */
    Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        Optional<float3> optional_wo = sample(wi, rng);

        if (let wo = optional_wo) {
            float p = pdf(wi, wo);
            return ShadingFunctionSample(wo, p, eval(wi, wo, rng) / p);
        }

        return none;
    }
}

}
