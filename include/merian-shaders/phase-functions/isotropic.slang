#pragma once

#include "merian-shaders/utils/math-constants.slangh"

import "merian-shaders/utils/sampling.slang";

__exported import merian_shaders.phase_functions.phase_function;

namespace merian {

float3 phase_isotropic_sample(float2 random) {
    return sample_sphere(random);
}

float phase_isotropic_pdf() {
    return MERIAN_INV_FOUR_PI;
}

float3 phase_isotropic_eval() {
    return MERIAN_INV_FOUR_PI;
}

struct IsotropicPhaseFunction : PhaseFunction {

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        return MERIAN_INV_FOUR_PI;
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        return sample_sphere(rng.next_float2());
    }

    override Optional<PhaseFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        return PhaseFunctionSample(sample_sphere(rng.next_float2()), MERIAN_INV_FOUR_PI, 1.0f);
    }

    float pdf(const float3 wi, const float3 wo) {
        return MERIAN_INV_FOUR_PI;
    }
}

}
