__exported import merian_shaders.shading.phase_functions.mie_approx;
__exported import merian_shaders.shading.phase_functions.isotropic;

namespace merian {

// A medium with constant coefficients. Implementations are selected statically, so Vacuum folds
// the medium out of a renderer entirely rather than being branched around at runtime.
[anyValueSize(48)]
interface HomogeneousVolume {
    associatedtype PhaseFunctionImpl : PhaseFunction;

    // Beer-Lambert attenuation over a segment of length t.
    float3 transmittance(const float t);

    // Scattering coefficient per unit distance.
    float3 get_mu_s();

    // Scalar extinction driving distance sampling. The estimator divides by the pdf built from
    // this, so a colored extinction stays unbiased.
    float sampling_mu_t();

    // Optical depth saturates beyond this distance, so an unbounded medium cannot extinguish the
    // environment.
    float get_max_distance();

    // Relative index of refraction (incident / transmitted) of the boundary enclosing this volume.
    float get_eta();

    PhaseFunctionImpl get_phase_function();
}

struct Vacuum : HomogeneousVolume {
    typealias PhaseFunctionImpl = IsotropicPhaseFunction;

    float3 transmittance(const float t) {
        return float3(1.0f);
    }
    float3 get_mu_s() {
        return float3(0.0f);
    }
    float sampling_mu_t() {
        return 0.0f;
    }
    float get_max_distance() {
        return 1e7f;
    }
    float get_eta() {
        return 1.0f;
    }
    IsotropicPhaseFunction get_phase_function() {
        return IsotropicPhaseFunction();
    }
}

// Scattering medium with the Mie phase function fit. The fit coefficients are precomputed on the
// host from the droplet diameter.
struct Fog : HomogeneousVolume {
    typealias PhaseFunctionImpl = MieApproxPhaseFunction;

    float3 mu_t;
    float3 mu_s;
    float max_distance;
    MieApproxPhaseFunction phase;

    float3 transmittance(const float t) {
        return exp(-min(t, max_distance) * mu_t);
    }
    float3 get_mu_s() {
        return mu_s;
    }
    float sampling_mu_t() {
        return (mu_t.r + mu_t.g + mu_t.b) / 3.0f;
    }
    float get_max_distance() {
        return max_distance;
    }
    float get_eta() {
        return 1.0f;
    }
    MieApproxPhaseFunction get_phase_function() {
        return phase;
    }
}

// Medium behind a refractive boundary. Materials that only absorb leave mu_s at 0.
struct IsotropicVolume : HomogeneousVolume {
    typealias PhaseFunctionImpl = IsotropicPhaseFunction;

    float3 mu_t;
    float3 mu_s;
    float eta;

    __init(const float3 mu_t, const float3 mu_s, const float eta) {
        this.mu_t = mu_t;
        this.mu_s = mu_s;
        this.eta = eta;
    }

    float3 transmittance(const float t) {
        return exp(-t * mu_t);
    }
    float3 get_mu_s() {
        return mu_s;
    }
    float sampling_mu_t() {
        return (mu_t.r + mu_t.g + mu_t.b) / 3.0f;
    }
    float get_max_distance() {
        return 1e7f;
    }
    float get_eta() {
        return eta;
    }
    IsotropicPhaseFunction get_phase_function() {
        return IsotropicPhaseFunction();
    }
}

} // namespace merian
