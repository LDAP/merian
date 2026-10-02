__exported import merian_shaders.distributions.distribution;
__exported import merian_shaders.shading.bsdfs.bsdf;
__exported import merian_shaders.shading.shading_function;
__exported import merian_shaders.utils.frame;
__exported import merian_shaders.utils.random;
__exported import merian_shaders.colors.colorspaces;

namespace merian {

static const int SCATTER_MODE_OFF = 0;
static const int SCATTER_MODE_MIS = 1;
static const int SCATTER_MODE_RIS = 2;

struct RenderFrameInfo {
    float3 cam_pos;
    float3 prev_cam_pos;
    float time;
    float time_diff;
    uint index;
}

// One bit per followed surface below a leading 1, set where it transmitted. Rough once a surface
// did not scatter singularly or more than seven were followed.
static const uint8_t SPECULAR_CHAIN_EMPTY = 1;
static const uint8_t SPECULAR_CHAIN_ROUGH = 0;

uint8_t specular_chain_append(const uint8_t chain, const bool singular, const bool transmission) {
    return chain == SPECULAR_CHAIN_ROUGH || chain >= uint8_t(0x80) || !singular
               ? SPECULAR_CHAIN_ROUGH
               : uint8_t((uint(chain) << 1u) | (transmission ? 1u : 0u));
}

// Plain data: a material here would drag the material existential into every guiding method.
struct PathVertex {
    float3 pos;
    float3 prev_pos;
    OrthonormalFrame shading_frame;
    float alpha;
    uint16_t bounce; // 1 is the first scattering vertex
    uint16_t sample_index;
    uint2 pixel;
    bool in_volume;
    // turned to the shading normal's side
    float3 geo_normal;
    bool transmissive;

    property float3 normal {
        get {
            return shading_frame.get_normal();
        }
    }
}

// Live across the continuation ray: keep it narrow.
struct PathSegment {
    float3 pos;
    float3 normal;
    uint16_t bounce;
    // the surfaces followed to reach next_pos
    uint8_t chain;

    float3 wo;
    // over every technique that could have generated wo
    float pdf;
    // throughput from pos to next_pos, the surfaces followed on the way included
    float3 weight;
    bool delta;
    bool is_last;

    // Set only where a continuation ray was traced. A ray that left the scene is traced.
    bool traced;
    float3 next_pos;
    // where the light along wo appears to come from, behind every surface the path followed
    float3 aim_pos;
    float3 next_normal;
    half3 next_mv;
    // emitted at next_pos back along the path
    float3 incident;
    // the share of an emitter's incident radiance a guiding method learns
    float direct_scale;
}

interface GuidingDistribution<T : IFloat> : Distribution<T> {
    bool is_valid();

    // Scales the share the integrator gives this distribution; may exceed 1.
    float get_confidence();
}

interface GuidingDebugModel {
    float4 debug_view(const RenderFrameInfo frame_info,
                      const PathVertex vertex,
                      inout RandomGenerator rng);
}

interface GuidingModel : GuidingDebugModel {
    associatedtype PathState;
    associatedtype DistributionImpl : GuidingDistribution<float3>;
    // the only state that survives the continuation ray
    associatedtype UpdateRecord;

    PathState begin_path();

    DistributionImpl get_distribution(inout PathState state, const RenderFrameInfo frame_info,
                                      const PathVertex vertex, inout RandomGenerator rng);

    // Before the continuation ray: the segment carries only pos, normal, bounce, wo, pdf and
    // delta. A zero pdf means the vertex generated no direction.
    UpdateRecord record_scatter(const DistributionImpl dist, const PathSegment segment,
                                const bool guided, inout RandomGenerator rng);

    // Returns radiance the integrator adds to complete the path at the next vertex.
    float3 record_segment(inout PathState state, const RenderFrameInfo frame_info,
                          const UpdateRecord record, const PathSegment segment,
                          inout RandomGenerator rng);
}

struct RayVertex {
    float3 origin;
    float3 direction;
    float max_distance;
    uint2 pixel;
    uint16_t sample_index;
}

struct RaySegment {
    float distance;
    // luminance of the sample's contribution
    float weight;
}

interface DistanceGuidingModel {
    associatedtype PathState;
    associatedtype DistributionImpl : GuidingDistribution<float>;
    associatedtype UpdateRecord;

    PathState begin_path();

    DistributionImpl get_distribution(inout PathState state, const RenderFrameInfo frame_info,
                                      const RayVertex vertex, inout RandomGenerator rng);

    UpdateRecord record_scatter(const DistributionImpl dist, const float distance,
                                const bool guided, inout RandomGenerator rng);

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

    float4 debug_view(const RenderFrameInfo frame_info,
                      const PathVertex vertex,
                      inout RandomGenerator rng) {
        return float4(0);
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
                         const float distance,
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

struct ScatterSample {
    float3 wo;
    // f(wi, wo) * cos; for a delta lobe the sample weight
    float3 f_cos;
    // mixture density at wo; for a delta lobe the probability of having drawn from the BSDF
    float pdf;
    // pdf, or 1/W for a resampled draw
    float realized_pdf;
    bool delta;
    bool guided;
    BSDFLobe lobe;
}

static const int GUIDING_DIRECT_FULL = 0;
static const int GUIDING_DIRECT_MIS = 1;
static const int GUIDING_DIRECT_NONE = 2;

extern static const float merian_guiding_share = 0.5;
extern static const bool merian_guiding_scale_with_alpha = true;
extern static const float merian_guiding_alpha_threshold = 0.05;
extern static const int merian_guiding_direct_target = GUIDING_DIRECT_FULL;

float guiding_share<G : GuidingDistribution<float3>>(const G guiding, const float alpha) {
    if (!guiding.is_valid() || alpha < merian_guiding_alpha_threshold) {
        return 0.f;
    }
    const float share =
        merian_guiding_scale_with_alpha ? merian_guiding_share * alpha : merian_guiding_share;
    return clamp(share * guiding.get_confidence(), 0.f, 1.f);
}

float guiding_direct_scale(const float mis_weight) {
    switch (merian_guiding_direct_target) {
    case GUIDING_DIRECT_MIS:
        return mis_weight;
    case GUIDING_DIRECT_NONE:
        return 0.f;
    default:
        return 1.f;
    }
}

// The shading function and the guiding distribution are passed per call so that neither stays
// live across the continuation ray.
struct ScatterMixture {
    float p_guiding;

    __init(const float share) {
        p_guiding = share;
    }

    // The guiding term comes first: it consumes the candidates before the material switch.
    float pdf<B : ShadingFunction, G : GuidingDistribution<float3>>(const B bsdf,
                                                                    const G guiding,
                                                                    const float3 wi_local,
                                                                    const float3 wo_local,
                                                                    const float3 wo_world) {
        const float p_guided = p_guiding > 0.f ? p_guiding * guiding.pdf(wo_world) : 0.f;
        return p_guided + (1.f - p_guiding) * bsdf.pdf(wi_local, wo_local);
    }

    Optional<ScatterSample>
    sample<B : ShadingFunction, G : GuidingDistribution<float3>>(const B bsdf,
                                                                 const G guiding,
                                                                 const OrthonormalFrame frame,
                                                                 const float3 wi_local,
                                                                 inout RandomGenerator rng) {
        ScatterSample s;
        s.delta = false;
        s.lobe = BSDFLobe::None;
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
            s.lobe = bs.value.lobe;
            if (bsdf_lobe_is_delta(bs.value.lobe)) {
                s.f_cos = bs.value.weight;
                s.pdf = 1.f - p_guiding;
                s.realized_pdf = s.pdf;
                s.delta = true;
                return s;
            }
        }

        s.f_cos = bsdf.eval(wi_local, wo_local, rng);
        s.pdf = pdf(bsdf, guiding, wi_local, wo_local, s.wo);
        s.realized_pdf = s.pdf;
        return s;
    }

    Optional<ScatterSample>
    sample_ris<B : ShadingFunction, G : GuidingDistribution<float3>>(const B bsdf,
                                                                     const G guiding,
                                                                     const OrthonormalFrame frame,
                                                                     const int candidates,
                                                                     const float3 wi_local,
                                                                     inout RandomGenerator rng) {
        if (!(p_guiding > 0.f)) {
            return sample(bsdf, guiding, frame, wi_local, rng);
        }

        ScatterSample selected;
        float target = 0.f;
        float weight_sum = 0.f;

        for (int i = 0; i < candidates; i++) {
            if (let c = sample(bsdf, guiding, frame, wi_local, rng)) {
                if (c.delta) {
                    return c;
                }
                const float t = yuv_luminance(c.f_cos);
                const float w = c.pdf > 0.f ? t / c.pdf : 0.f;
                weight_sum += w;
                if (w > 0.f && rng.next_float() * weight_sum < w) {
                    selected = c;
                    target = t;
                }
            }
        }

        if (!(weight_sum > 0.f)) {
            return none;
        }
        selected.f_cos *= selected.pdf * weight_sum / (target * candidates);
        selected.realized_pdf = target * candidates / weight_sum;
        return selected;
    }
}

} // namespace merian
