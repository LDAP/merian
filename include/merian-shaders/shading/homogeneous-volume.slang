__exported import merian_shaders.shading.phase_functions.mie_approx;
__exported import merian_shaders.shading.phase_functions.isotropic;

namespace merian {

// A medium with constant coefficients. Implementations are selected statically, so Vacuum folds
// the medium out of a renderer entirely rather than being branched around at runtime.
interface HomogeneousVolume {
    associatedtype PhaseFunctionImpl : PhaseFunction;

    float3 transmittance(const float t);

    float3 get_mu_s();

    // The estimator divides by a pdf built from this, so a colored extinction stays unbiased.
    float sampling_mu_t();

    // Optical depth saturates beyond this, so an unbounded medium cannot extinguish the
    // environment.
    float get_max_distance();

    // Relative index of refraction (incident / transmitted) of the enclosing boundary.
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

// Materials that only absorb leave mu_s at 0.
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
