#pragma once

#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.utils.sampling;

__exported import phase_function;

namespace merian {

float3 phase_isotropic_sample(inout RandomGenerator rng) {
    return sample_sphere(rng);
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
        return sample_sphere(rng);
    }

    override Optional<ShadingFunctionSample> sample_eval(const float3 wi,
                                                         inout RandomGenerator rng) {
        return ShadingFunctionSample(sample_sphere(rng), MERIAN_INV_FOUR_PI, 1.0f);
    }

    float pdf(const float3 wi, const float3 wo) {
        return MERIAN_INV_FOUR_PI;
    }
}

}
