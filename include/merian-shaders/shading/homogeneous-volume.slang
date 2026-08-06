__exported import merian_shaders.shading.phase_functions.mie_approx;

namespace merian {

// A medium with constant coefficients. mu_t == 0 is vacuum and every renderer is expected to skip
// the medium entirely in that case.
struct HomogeneousVolume {
    float3 mu_t; // extinction (absorption + scattering) per unit distance
    float3 mu_s; // scattering per unit distance; mu_t - mu_s is the absorption

    // Relative index of refraction (incident / transmitted) of the boundary enclosing this volume.
    float eta;

    // Droplet diameter driving the Mie phase function fit, which holds for 5 - 50 um.
    float particle_size_um;

    // Optical depth saturates beyond this distance, so an unbounded exterior medium cannot
    // extinguish the environment.
    float max_distance;

    // vacuum
    __init() {
        mu_t = float3(0.0f);
        mu_s = float3(0.0f);
        eta = 1.0f;
        particle_size_um = 25.0f;
        max_distance = 1e7f;
    }

    __init(const float3 mu_t,
           const float3 mu_s,
           const float eta,
           const float particle_size_um,
           const float max_distance) {
        this.mu_t = mu_t;
        this.mu_s = mu_s;
        this.eta = eta;
        this.particle_size_um = particle_size_um;
        this.max_distance = max_distance;
    }

    // purely absorbing
    __init(const float3 absorption, const float eta) {
        this.mu_t = absorption;
        this.mu_s = float3(0.0f);
        this.eta = eta;
        this.particle_size_um = 25.0f;
        this.max_distance = 1e7f;
    }

    // ------------------------------------------

    bool is_vacuum() {
        return all(mu_t == 0.0f);
    }

    // Beer-Lambert over a segment of length t.
    float3 transmittance(const float t) {
        return exp(-min(t, max_distance) * mu_t);
    }

    MieApproxPhaseFunction get_phase_function() {
        return MieApproxPhaseFunction(particle_size_um);
    }

    // Scalar extinction driving distance sampling. The estimator divides by the pdf built from
    // this, so a colored mu_t stays unbiased.
    float sampling_mu_t() {
        return (mu_t.r + mu_t.g + mu_t.b) / 3.0f;
    }
}

} // namespace merian
