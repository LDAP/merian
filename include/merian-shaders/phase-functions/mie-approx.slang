#pragma once

__exported import merian_shaders.phase_functions.phase_function;

import "merian-shaders/utils/sampling.slang";

import merian_shaders.phase_functions.henyey_greenstein;
import merian_shaders.phase_functions.draine;

namespace merian {

// An implementation of Jendersie and d'Eon (2023). An Approximate Mie Scattering Function for Fog
// and Cloud Rendering SIGGRAPH 2023 Talks https://doi.org/10.1145/3587421.3595409
//
// This is a mixture of the HenyeyGreenstein and Draine phase functions.
struct MieApproxPhaseFunction : PhaseFunction {

    // cos_theta is the cosine if the angle between the old particle direction and the new particle
    // direction (cosine of deflection). cos_theta = dot(prev_dir, new_dir) = dot(-wi, wo)
    static float eval(const float cos_theta,
                      const float hg_g,
                      const float d_g,
                      const float d_a,
                      const float w_d) {
        return lerp(HenyeyGreensteinPhaseFunction.eval(cos_theta, hg_g),
                    DrainePhaseFunction.eval(cos_theta, d_g, d_a), w_d);
    }

    // Sample an exact deflection cosine, i.e. dot(prev_dir, next_dir) = dot(-wi, wo)
    //   xi = a uniform random real in [0,1]
    static float
    sample(const float2 xi, const float hg_g, const float d_g, const float d_a, const float w_d) {
        if (xi.x < w_d) {
            return DrainePhaseFunction.sample(xi.y, d_g, d_a);
        } else {
            return HenyeyGreensteinPhaseFunction.sample(xi.y, hg_g);
        }
    }

    // wi points away from the sampling point, i.e -wi is the particle direction.
    static float3 sample(inout RandomGenerator rng,
                         const float3 wi,
                         const float hg_g,
                         const float d_g,
                         const float d_a,
                         const float w_d) {
        const float deflection_cos = sample(rng.next_float2(), hg_g, d_g, d_a, w_d);
        return sample_cone2(wi, deflection_cos, rng.next_float());
    }

    // -----------------------------------------------

    // the Henyey Greenstein anisotropy parameter
    float hg_g;
    // the Draine anisotropy parameter
    float d_g;
    // the Draine alpha parameter
    float d_a;
    // mixture weight (1 means use only draine)
    float w_d;

    __init(const float hg_g, const float d_g, const float d_a, const float w_d) {
        this.hg_g = hg_g;
        this.d_g = d_g;
        this.d_a = d_a;
        this.w_d = w_d;
    }

    // fit suitable for 5 < d < 50 micrometer.
    __init(const float particle_size_um) {
        this.hg_g = exp(-0.0990567 / (particle_size_um - 1.67154));
        this.d_g = exp(-2.20679 / (particle_size_um + 3.91029) - 0.428934);
        this.d_a = exp(3.62489 - 8.29288 / (particle_size_um + 5.52825));
        this.w_d = exp(-0.599085 / (particle_size_um - 0.641583) - 0.665888);
    }

    // cos_theta = dot(prev_dir, new_dir) = dot(-wi, wo)
    float eval(float cos_theta) {
        return eval(cos_theta, hg_g, d_g, d_a, w_d);
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        return eval(dot(-wi, wo));
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        return sample(rng, wi, hg_g, d_g, d_a, w_d);
    }

    override Optional<PhaseFunctionSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
        const float wo = sample(rng, wi, hg_g, d_g, d_a, w_d);
        return PhaseFunctionSample(wo, pdf(wi, wo), 1.0f);
    }

    float pdf(const float3 wi, const float3 wo) {
        return eval(dot(-wi, wo));
    }
}

}
