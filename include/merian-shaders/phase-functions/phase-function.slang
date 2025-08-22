#pragma once

__exported import "merian-shaders/utils/random.slang";

namespace merian {

struct PhaseFunctionSample {
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

interface PhaseFunction {

    /** Evaluates the PhaseFunction. The incident and outgoing direction point away from the shading
     * location.
     *
     *  Returns in solid angle: f(wi, wo)
     */
    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng);

    // -----------------------------

    /** Samples the PhaseFunction. The incident and outgoing direction point away from the shading
     * location.
     *
     * Returns an outgoing direction wo.
     */
    Optional<float3> sample(const float3 wi, inout RandomGenerator rng);

    /** Samples and evaluates the PhaseFunction. The incident and outgoing direction point away from
     * the shading location.
     *
     */
    Optional<PhaseFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        Optional<float3> optional_wo = sample(wi, rng);

        if (let wo = optional_wo) {
            float p = pdf(wi, wo);
            return PhaseFunctionSample(wo, p, eval(wi, wo, rng) / p);
        }

        return none;
    }

    // -----------------------------

    /** Evaluates the pdf of the sample function of this PhaseFunction. The incident and outgoing
     * direction point away from the shading location.
     *
     *  Returns in solid angle.
     */
    float pdf(const float3 wi, const float3 wo);
}

}
