#pragma once

#include "merian-shaders/utils/math-constants.slangh"

import "merian-shaders/utils/sampling.slang";

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

}
