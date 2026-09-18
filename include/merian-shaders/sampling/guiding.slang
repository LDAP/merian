__exported import merian_shaders.distributions.distribution;
__exported import merian_shaders.shading.bsdfs.bsdf;
__exported import merian_shaders.shading.shading_function;
__exported import merian_shaders.utils.frame;
__exported import merian_shaders.utils.random;
__exported import merian_shaders.colors.colorspaces;

namespace merian {

// How the integrator turns the mixture into one direction.
static const int SCATTER_MODE_MIS = 0;
static const int SCATTER_MODE_RIS = 1;

// When, and from where, the current frame is rendered; the values are wave-uniform.
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
    // GGX alpha in [0, 1]: 0 is a perfect mirror, 1 ideal diffuse
    float alpha;
    uint16_t bounce; // 1 is the first scattering vertex
    uint16_t sample_index;
    // the pixel the path was started from
    uint2 pixel;
    // a medium vertex has no surface: the frame is built around the view direction and the lobe
    // width is the phase function's
    bool in_volume;
    // The geometric normal, turned to the shading frame's side. A direction below it would start
    // the continuation ray inside the geometry.
    float3 geo_normal;
    // The surface scatters to its far side, so a direction below either normal is not dead there.
    bool transmissive;

    property float3 normal {
        get {
            return shading_frame.get_normal();
        }
    }
}

/* The resolved continuation of a vertex.
 *
 * Every field sits in registers across the continuation ray, where the path tracer is already at
 * the spilling threshold: keep it narrow.
 */
struct PathSegment {
    // the vertex the ray started from
    float3 pos;
    float3 normal;
    uint16_t bounce;

    float3 wo; // in worldspace
    // density that generated wo, over every technique that could have
    float pdf;
    // f(wi, wo) * dot(wo, n) divided by that density
    float3 weight;
    // no technique other than the BSDF could have generated wo
    bool delta;
    // no vertex will be scattered from beyond this segment
    bool is_last;

    // set only where a continuation ray was traced:

    // False where the surface refused the direction. A ray that left the scene did trace: it
    // reached the environment.
    bool traced;
    // the vertex the ray ended on
    float3 next_pos;
    // Where the light along wo appears to come from: behind every surface the path was followed
    // through, at the distance the whole way there. Equal to next_pos where nothing was followed.
    float3 aim_pos;
    float3 next_normal;
    // worldspace velocity of the geometry at next_pos
    half3 next_mv;
    // radiance arriving from the next vertex
    float3 incident;
    // what a guiding method learns of that radiance, after the direct light policy
    float direct_scale;
}

// What a guiding method offers at one vertex: directions in worldspace (solid angle) for a
// scattering vertex, distances along a ray for a medium. It covers the guiding lobes only; the
// integrator mixes them with the analytic technique it would have sampled otherwise.
interface GuidingDistribution<T : IFloat> : Distribution<T> {

    // False where nothing was found; the integrator then samples the BSDF alone.
    bool is_valid();

    // Scales the share the integrator gives this distribution; may exceed 1.
    float get_confidence();
}

// What a method can show of its own state, for one point on screen.
interface GuidingDebugModel {
    bool debug_view(const int selector, const RenderFrameInfo frame_info, const PathVertex vertex,
                    inout RandomGenerator rng, out float4 color);
}

// A directional guiding method: a resource-holding object in the shader binding that hands out
// one GuidingDistribution per vertex and learns from the continuation.
interface GuidingModel : GuidingDebugModel {
    // Carried along a path, for methods that reuse what a lookup found one vertex earlier.
    associatedtype PathState;
    associatedtype DistributionImpl : GuidingDistribution<float3>;
    // Everything record_segment needs, and nothing else: it is the only thing that survives the
    // continuation ray.
    associatedtype UpdateRecord;

    PathState begin_path();

    DistributionImpl get_distribution(inout PathState state, const RenderFrameInfo frame_info,
                                      const PathVertex vertex, inout RandomGenerator rng);

    /* Narrows the distribution down to what the update needs, once the direction is known.
     *
     * Called before the continuation ray is traced, so that the distribution itself dies there.
     * The segment carries the direction but not yet what came back along it: only pos, wo, pdf
     * and the delta flag are set. A zero pdf means the vertex generated no direction at all.
     */
    UpdateRecord record_scatter(const DistributionImpl dist, const PathSegment segment,
                                const bool guided, inout RandomGenerator rng);

    /* Learns from the resolved continuation.
     *
     * Returns radiance that completes the path at the next vertex, or float3(0) where the method
     * has no such estimate. The integrator adds it in place of continuing.
     */
    float3 record_segment(inout PathState state, const RenderFrameInfo frame_info,
                          const UpdateRecord record, const PathSegment segment,
                          inout RandomGenerator rng);
}

/* The primary ray a medium sits on.
 *
 * Distance guiding is per pixel along one ray, so unlike a scattering vertex there is no frame and
 * no lobe width: the distribution is over the distance travelled before scattering.
 */
struct RayVertex {
    float3 origin;
    float3 direction;
    // how far the medium reaches along the ray
    float max_distance;
    uint2 pixel;
    uint16_t sample_index;
}

// The resolved scattering event of a RayVertex.
struct RaySegment {
    float distance;
    // what the sample contributed at that distance
    float weight;
}

// A distance guiding method: the same shape as GuidingModel, over distances rather than directions.
interface DistanceGuidingModel {
    associatedtype PathState;
    associatedtype DistributionImpl : GuidingDistribution<float>;
    associatedtype UpdateRecord;

    PathState begin_path();

    DistributionImpl get_distribution(inout PathState state, const RenderFrameInfo frame_info,
                                      const RayVertex vertex, inout RandomGenerator rng);

    // Narrows the distribution down to what the update needs, once the distance is known.
    UpdateRecord record_scatter(const DistributionImpl dist, const float wo, const bool guided,
                                inout RandomGenerator rng);

    // Learns from the resolved scattering event.
    void record_segment(inout PathState state, const RenderFrameInfo frame_info,
                        const UpdateRecord record, const RaySegment segment,
                        inout RandomGenerator rng);
}

struct NullGuidingDistribution : GuidingDistribution<float3> {
    float pdf(const float3 s) {
        return 0.f;
    }
    float3 sample(inout RandomGenerator rng) {
        return float3(0, 0, 1);
    }
    DistributionSample<float3> sample_eval(inout RandomGenerator rng) {
        return {float3(0, 0, 1), 0.f};
    }
    bool is_valid() {
        return false;
    }
    float get_confidence() {
        return 1.f;
    }
}

struct NullGuidingModel : GuidingModel {
    struct State {
    }

    typealias PathState = State;
    typealias DistributionImpl = NullGuidingDistribution;
    typealias UpdateRecord = State;

    State begin_path() {
        return {};
    }

    NullGuidingDistribution get_distribution(inout State state,
                                             const RenderFrameInfo frame_info,
                                             const PathVertex vertex,
                                             inout RandomGenerator rng) {
        return {};
    }

    State record_scatter(const NullGuidingDistribution dist,
                         const PathSegment segment,
                         const bool guided,
                         inout RandomGenerator rng) {
        return {};
    }

    float3 record_segment(inout State state,
                          const RenderFrameInfo frame_info,
                          const State record,
                          const PathSegment segment,
                          inout RandomGenerator rng) {
        return float3(0);
    }

    bool debug_view(const int selector,
                    const RenderFrameInfo frame_info,
                    const PathVertex vertex,
                    inout RandomGenerator rng,
                    out float4 color) {
        color = float4(0);
        return false;
    }
}

struct NullDistanceGuidingDistribution : GuidingDistribution<float> {
    float pdf(const float s) {
        return 0.f;
    }
    float sample(inout RandomGenerator rng) {
        return 0.f;
    }
    DistributionSample<float> sample_eval(inout RandomGenerator rng) {
        return {0.f, 0.f};
    }
    bool is_valid() {
        return false;
    }
    float get_confidence() {
        return 1.f;
    }
}

struct NullDistanceGuidingModel : DistanceGuidingModel {
    struct State {
    }

    typealias PathState = State;
    typealias DistributionImpl = NullDistanceGuidingDistribution;
    typealias UpdateRecord = State;

    State begin_path() {
        return {};
    }

    NullDistanceGuidingDistribution get_distribution(inout State state,
                                                     const RenderFrameInfo frame_info,
                                                     const RayVertex vertex,
                                                     inout RandomGenerator rng) {
        return {};
    }

    State record_scatter(const NullDistanceGuidingDistribution dist,
                         const float wo,
                         const bool guided,
                         inout RandomGenerator rng) {
        return {};
    }

    void record_segment(inout State state,
                        const RenderFrameInfo frame_info,
                        const State record,
                        const RaySegment segment,
                        inout RandomGenerator rng) {}
}

// One draw from the scatter mixture.
struct ScatterSample {
    float3 wo; // in worldspace
    // f(wi, wo) * dot(wo, n); for a delta lobe the sample weight, which already divides by the
    // lobe's own probability
    float3 f_cos;
    // density of the mixture at wo; for a delta lobe the probability of the strategy that
    // produced it, so that f_cos / pdf is the throughput factor either way
    float pdf;
    // no other technique can generate this direction
    bool delta;
    bool guided;
}

// What a guiding method learns from a vertex that ended on a light.
static const int GUIDING_DIRECT_FULL = 0; // the emission as the continuation found it
static const int GUIDING_DIRECT_MIS = 1;  // only the share the scatter technique pays for
static const int GUIDING_DIRECT_NONE = 2; // nothing

// How the integrator splits its one scatter sample with a guiding method.
extern static const float merian_guiding_share = 0.5;
extern static const bool merian_guiding_scale_with_alpha = true;
// below this lobe width the guiding lobes are broader than the shading function itself
extern static const float merian_guiding_alpha_threshold = 0.05;
extern static const int merian_guiding_direct_target = GUIDING_DIRECT_FULL;

// The share of the scatter samples a guiding distribution gets at a vertex of this width.
float guiding_share<G : GuidingDistribution<float3>>(const G guiding, const float alpha) {
    if (!guiding.is_valid() || alpha < merian_guiding_alpha_threshold) {
        return 0.f;
    }
    const float share =
        merian_guiding_scale_with_alpha ? merian_guiding_share * alpha : merian_guiding_share;
    return clamp(share * guiding.get_confidence(), 0.f, 1.f);
}

// One-sample MIS between a shading function and a guiding distribution: a BSDF at a surface, a
// phase function in a medium. Both are passed in per call so that neither stays live across the
// continuation ray.
struct ScatterMixture {
    // 0 where the guiding distribution has nothing to offer
    float p_guiding;

    __init(const float share) {
        p_guiding = share;
    }

    // The guiding term comes first on purpose: it consumes the candidates, and the material pdf
    // below is an existential switch that would otherwise raise the peak by the whole candidate
    // set.
    float pdf<B : ShadingFunction, G : GuidingDistribution<float3>>(const B bsdf,
                                                                    const G guiding,
                                                                    const float3 wi_local,
                                                                    const float3 wo_local,
                                                                    const float3 wo_world) {
        const float p_guided = p_guiding > 0.f ? p_guiding * guiding.pdf(wo_world) : 0.f;
        return p_guided + (1.f - p_guiding) * bsdf.pdf(wi_local, wo_local);
    }

    // Draws a direction in worldspace; none where the shading function refused to scatter.
    Optional<ScatterSample>
    sample<B : ShadingFunction, G : GuidingDistribution<float3>>(const B bsdf,
                                                                 const G guiding,
                                                                 const OrthonormalFrame frame,
                                                                 const float3 wi_local,
                                                                 inout RandomGenerator rng) {
        ScatterSample s;
        s.delta = false;
        s.guided = p_guiding > 0.f && rng.next_float() < p_guiding;

        float3 wo_local;
        if (s.guided) {
            s.wo = guiding.sample(rng);
            wo_local = frame.to_local(s.wo);
        } else {
            const Optional<ShadingFunctionSample> bs = bsdf.sample_eval(wi_local, rng);
            if (!bs.hasValue) {
                return none;
            }
            wo_local = bs.value.wo;
            s.wo = frame.to_world(wo_local);
            if (bsdf_lobe_is_delta(bs.value.lobe)) {
                // a delta direction has no density under the guiding lobe, so all that is left of
                // the mixture is the chance of having drawn from the BSDF at all
                s.f_cos = bs.value.weight;
                s.pdf = 1.f - p_guiding;
                s.delta = true;
                return s;
            }
        }

        s.f_cos = bsdf.eval(wi_local, wo_local, rng);
        s.pdf = pdf(bsdf, guiding, wi_local, wo_local, s.wo);
        return s;
    }

    /* Resampled importance sampling over the same two techniques.
     *
     * Draws `candidates` directions from the mixture, weighs each by how much the shading
     * function makes of it, and keeps one. Unlike the single draw above it can pass over a
     * direction the shading function cannot use, at the cost of the extra evaluations; it still
     * traces one ray. The target is the shading function alone: neither the guiding lobes nor the
     * caches behind them carry radiance per direction, so there is nothing to weigh it against.
     */
    Optional<ScatterSample>
    sample_ris<B : ShadingFunction, G : GuidingDistribution<float3>>(const B bsdf,
                                                                     const G guiding,
                                                                     const OrthonormalFrame frame,
                                                                     const int candidates,
                                                                     const float3 wi_local,
                                                                     inout RandomGenerator rng) {
        // without guiding lobes the mixture is the shading function, and resampling it against
        // itself gains nothing
        if (!(p_guiding > 0.f)) {
            return sample(bsdf, guiding, frame, wi_local, rng);
        }

        ScatterSample selected;
        float target = 0.f;
        float selected_pdf = 0.f;
        float weight_sum = 0.f;

        for (int i = 0; i < candidates; i++) {
            ScatterSample c;
            c.delta = false;
            c.guided = rng.next_float() < p_guiding;

            float3 wo_local;
            if (c.guided) {
                c.wo = guiding.sample(rng);
                wo_local = frame.to_local(c.wo);
            } else {
                const Optional<ShadingFunctionSample> bs = bsdf.sample_eval(wi_local, rng);
                if (!bs.hasValue) {
                    continue;
                }
                wo_local = bs.value.wo;
                c.wo = frame.to_world(wo_local);
                if (bsdf_lobe_is_delta(bs.value.lobe)) {
                    // a delta direction has no density to resample against, so it leaves the pool
                    // the way the single draw takes it
                    c.f_cos = bs.value.weight;
                    c.pdf = 1.f - p_guiding;
                    c.delta = true;
                    return c;
                }
            }

            c.f_cos = bsdf.eval(wi_local, wo_local, rng);
            const float q = pdf(bsdf, guiding, wi_local, wo_local, c.wo);
            const float t = yuv_luminance(c.f_cos);
            const float w = q > 0.f ? t / q : 0.f;
            weight_sum += w;
            if (w > 0.f && rng.next_float() * weight_sum < w) {
                selected = c;
                target = t;
                selected_pdf = q;
            }
        }

        if (!(weight_sum > 0.f)) {
            return none;
        }
        selected.f_cos *= selected_pdf * weight_sum / (target * candidates);
        selected.pdf = selected_pdf;
        return selected;
    }
}

} // namespace merian
