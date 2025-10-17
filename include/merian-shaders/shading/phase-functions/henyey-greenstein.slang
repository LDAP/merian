#pragma once

#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.utils.sampling;
import merian_shaders.utils.math;

__exported import phase_function;

namespace merian {

struct HenyeyGreensteinPhaseFunction : PhaseFunction {

    // cos_theta is the cosine if the angle between the old particle direction and the new particle
    // direction (cosine of deflection). cos_theta = dot(prev_dir, new_dir) = dot(-wi, wo)
    static float eval(float cos_theta, float g) {
        float g2 = merian::pow2(g);
        float denom = (1 + g2 - 2 * g * cos_theta);
        return MERIAN_INV_FOUR_PI * (1.0 - g2) / (denom * sqrt(denom));
    }

    // Sample an exact deflection cosine, i.e. dot(prev_dir, next_dir) = dot(-wi, wo)
    //   xi = a uniform random real in [0,1]
    static float sample(const float xi, const float g) {
        if (abs(g) < 1e-3f) {
            return 1.f - 2.f * xi;
        }

        float g2 = merian::pow2(g);
        float t = (1 - g2) / (1 - g + 2 * g * xi);
        return (1 + g2 - t * t) / (2 * g);
    }

    // wi points away from the sampling point, i.e -wi is the particle direction.
    static float3 sample(inout RandomGenerator rng, const float3 wi, const float g) {
        const float deflection_cos = sample(rng.next_float(), g);
        return sample_cone2(wi, deflection_cos, rng.next_float());
    }

    // ----------------------------------------------------------

    float g; // HG shape / anisotropy parameter

    __init(const float g) {
        this.g = g;
    }

    // cos_theta = dot(prev_dir, new_dir) = dot(-wi, wo)
    float eval(float cos_theta) {
        return eval(cos_theta, g);
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        return eval(dot(-wi, wo));
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        return sample(rng, wi, g);
    }

    override Optional<ShadingFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        const float wo = sample(rng, wi, g);
        return ShadingFunctionSample(wo, pdf(wi, wo), 1.0f);
    }

    float pdf(const float3 wi, const float3 wo) {
        return eval(dot(-wi, wo));
    }
}

}
