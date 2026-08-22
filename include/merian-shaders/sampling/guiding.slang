__exported import merian_shaders.distributions.distribution;
__exported import merian_shaders.shading.bsdfs.bsdf;
__exported import merian_shaders.utils.frame;
__exported import merian_shaders.utils.random;

namespace merian {

// When, and from where, the current frame is rendered. Spatial guiding structures are
// camera-adaptive and age their records, so they need it; passed rather than stored, the values
// are wave-uniform.
struct RenderFrameInfo {
    float3 cam_pos;
    float3 prev_cam_pos;
    float time;
    float time_diff;
    uint index;
}

// What a guiding method is told at a vertex before the continuation ray is traced. Plain data: a
// material sample must never enter here, it would drag the material existential into every
// guiding method.
struct PathVertex {
    float3 pos;
    // where the vertex was one frame ago; guiding methods with temporal state look up here
    float3 prev_pos;
    OrthonormalFrame shading_frame;
    float roughness;
    uint16_t bounce; // 1 is the first scattering vertex
    uint16_t sample_index;

    property float3 normal {
        get { return shading_frame.get_normal(); }
    }
}

/* The resolved continuation of a vertex.
 *
 * Deliberately narrow, and it repeats the few vertex fields a guiding method still needs: every
 * field in here sits in registers across the traversal, and the path tracer is already at the
 * spilling threshold there.
 */
struct PathSegment {
    // the vertex the ray started from
    float3 pos;
    float3 normal;
    uint16_t bounce;

    float3 wo; // in worldspace
    // density that generated wo, over the whole scatter mixture
    float pdf;
    // f(wi, wo) * dot(wo, n), without the density
    float3 f_cos;
    // no vertex will be scattered from beyond this segment
    bool is_last;

    // --- only set where a continuation ray was traced ---

    // False where the surface refused the direction. A ray that left the scene did trace: it
    // reached the environment.
    bool traced;
    float3 next_pos;
    float3 next_normal;
    // worldspace velocity of the geometry at next_pos
    half3 next_mv;
    // radiance arriving from the next vertex
    float3 incident;
}

// The directional distribution a guiding method offers at one vertex. Densities are solid angle
// in worldspace and cover the guiding lobes only; the integrator mixes them with the BSDF.
interface GuidingDistribution : Distribution<float3> {

    // False where nothing was found; the integrator then samples the BSDF alone.
    bool is_valid();

    // Probability the integrator gives this distribution in the one-sample mixture.
    float get_selection_probability(const float roughness);
}

// A directional guiding method: a resource-holding object in the shader binding that hands out
// one GuidingDistribution per vertex and learns from the continuation.
interface GuidingModel {
    // Carried along a path, for methods that reuse what a lookup found one vertex earlier.
    associatedtype PathState;
    associatedtype DistributionImpl : GuidingDistribution;
    // Everything record_segment needs, and nothing else: it is the only thing that survives the
    // continuation ray, and the path tracer already spills there.
    associatedtype UpdateRecord;

    PathState begin_path();

    DistributionImpl get_distribution(inout PathState state,
                                  const RenderFrameInfo frame_info,
                                  const PathVertex vertex,
                                  inout RandomGenerator rng);

    /* Narrows the distribution down to what the update needs, once the direction is known.
     *
     * Called before the continuation ray is traced, so that the distribution itself dies there.
     */
    UpdateRecord record_scatter(const DistributionImpl dist, const bool guided);

    /* Learns from the resolved continuation.
     *
     * Returns radiance that completes the path at the next vertex, or float3(0) where the method
     * has no such estimate. The integrator adds it in place of continuing.
     */
    float3 record_segment(inout PathState state,
                      const RenderFrameInfo frame_info,
                      const UpdateRecord record,
                      const PathSegment segment,
                      inout RandomGenerator rng);
}

// --- no guiding ---

struct NullGuidingDistribution : GuidingDistribution {
    float pdf(const float3 s) { return 0.f; }
    float3 sample(inout RandomGenerator rng) { return float3(0, 0, 1); }
    DistributionSample<float3> sample_eval(inout RandomGenerator rng) {
        return {float3(0, 0, 1), 0.f};
    }
    bool is_valid() { return false; }
    float get_selection_probability(const float roughness) { return 0.f; }
}

struct NullGuidingModel : GuidingModel {
    struct State {}

    typealias PathState = State;
    typealias DistributionImpl = NullGuidingDistribution;
    typealias UpdateRecord = State;

    State begin_path() { return {}; }

    NullGuidingDistribution get_distribution(inout State state,
                                         const RenderFrameInfo frame_info,
                                         const PathVertex vertex,
                                         inout RandomGenerator rng) {
        return {};
    }

    State record_scatter(const NullGuidingDistribution dist, const bool guided) {
        return {};
    }

    float3 record_segment(inout State state,
                      const RenderFrameInfo frame_info,
                      const State record,
                      const PathSegment segment,
                      inout RandomGenerator rng) {
        return float3(0);
    }
}

// One-sample MIS between the BSDF and a guiding distribution. Neither is stored: the BSDF comes
// out of the material system as an existential, and the distribution is dead as soon as the
// direction is drawn — keeping either would hold it live across the continuation ray.
struct ScatterMixture {
    // 0 where the guiding distribution has nothing to offer
    float p_guiding;

    __init<G : GuidingDistribution>(const G guiding, const float roughness) {
        p_guiding = guiding.is_valid() ? guiding.get_selection_probability(roughness) : 0.f;
    }

    // Draws a direction in worldspace; none where the BSDF refused to scatter.
    Optional<float3> sample<B : BSDF, G : GuidingDistribution>(const B bsdf,
                                                               const G guiding,
                                                               const OrthonormalFrame frame,
                                                               const float3 wi_local,
                                                               inout RandomGenerator rng,
                                                               out bool guided) {
        guided = p_guiding > 0.f && rng.next_float() < p_guiding;
        if (guided) {
            return guiding.sample(rng);
        }
        if (let wo_local = bsdf.sample(wi_local, rng)) {
            return frame.to_world(wo_local);
        }
        return none;
    }

    // The guiding term comes first on purpose: it consumes the candidates, and the material pdf
    // below is an existential switch that would otherwise raise the peak by the whole candidate
    // set.
    float pdf<B : BSDF, G : GuidingDistribution>(const B bsdf,
                                                 const G guiding,
                                                 const float3 wi_local,
                                                 const float3 wo_local,
                                                 const float3 wo_world) {
        const float p_guided = p_guiding > 0.f ? p_guiding * guiding.pdf(wo_world) : 0.f;
        return p_guided + (1.f - p_guiding) * bsdf.pdf(wi_local, wo_local);
    }
}

} // namespace merian
