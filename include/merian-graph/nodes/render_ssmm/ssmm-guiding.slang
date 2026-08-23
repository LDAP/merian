#include "merian-graph/nodes/render_ssmm/ssmm_guiding.slangh"

__exported import merian_shaders.sampling.guiding;
import merian_shaders.gbuffer;
import merian_shaders.distributions.von_mises_fisher;
import merian_shaders.colors.colorspaces;

using merian;

namespace merian {
// neighbour fits resampled towards this vertex
extern static const int merian_ssmm_group_size = 5;
// standard deviation of the neighbourhood the resampling reaches into, in pixels
extern static const float merian_ssmm_reuse_radius = 15.0;
// probability the integrator gives the fitted lobe
extern static const float merian_ssmm_probability = 0.85;
// no guiding below this lobe width: the BSDF is far more peaked than any fitted lobe
extern static const float merian_ssmm_alpha_threshold = 0.05;
// samples folded into the exponentially weighted fit
extern static const uint merian_ssmm_max_n = 1024u;
extern static const float merian_ssmm_min_alpha = 0.01;
extern static const float merian_ssmm_prior_n = 0.2;
} // namespace merian

// Screen-space mixture models (Dittebrandt et al. 2023), ported from VKDT with the author's
// approval. One fitted lobe per pixel, carried over frames through the motion vectors; it
// describes the first scattering vertex only.
namespace merian {

bool ssmm_valid(const SSMCState s) {
    return s.sum_w > 0.0;
}

float3 ssmm_target_dir(const SSMCState s, const float3 x) {
    return normalize(s.sum_tgt / (s.sum_w > 0.0 ? s.sum_w : 1.0) - x);
}

// the vMF lobe (direction, concentration) the state stands for at x
float4 ssmm_vmf(const SSMCState s, const float3 x) {
    float r = s.sum_len / (s.sum_w > 0.0 ? s.sum_w : 1.0); // mean cosine in [0, 1]
    r = (s.N * s.N * r) / (s.N * s.N + merian_ssmm_prior_n);
    return float4(ssmm_target_dir(s, x), (3.0 * r - r * r * r) / (1.0 - r * r));
}

// maximum likelihood update, exponentially weighted
void ssmm_add(inout SSMCState s, const float3 x, const float w, const float3 dir, const float3 y) {
    s.N = min(s.N + 1, merian_ssmm_max_n);
    const float alpha = max(1.0 / s.N, merian_ssmm_min_alpha);
    s.sum_w = lerp(s.sum_w, w, alpha);
    s.sum_tgt = lerp(s.sum_tgt, w * y, alpha);
    float3 to = s.sum_len * ssmm_target_dir(s, x);
    to = lerp(to, w * dir, alpha);
    s.sum_len = length(to);
}

// How much a neighbour's fit is worth here: its own score, discounted where the surface it was
// fitted on does not match this one. SVGF-style normal and depth weights.
float ssmm_reuse_weight(const SSMCState s,
                        const ParameterBlock<GBuffer> gbuffer,
                        const float3 cam_pos,
                        const uint2 resolution,
                        const float3 pos,
                        const float3 normal,
                        const int2 tc) {
    const uint2 q = uint2(clamp(tc, int2(0), int2(resolution) - 1));
    const float w_n = pow(max(0.0, dot(normal, gbuffer.get_normal(q))), 64.0);
    const float w_d = exp(-abs(gbuffer.get_linear_z(q) - length(pos - cam_pos)) / 10.0);
    return s.f * w_n * w_d;
}

// The one fit resampled out of the neighbourhood, as a vMF lobe.
struct SSMMDistribution : GuidingDistribution<float3> {
    float4 vmf;
    // the fit the update continues from
    SSMCState selected;

    float pdf(const float3 s) {
        return vmf.w > 0.0 ? vmf_pdf_tokuyoshi(s, vmf.xyz, vmf.w) : 0.0;
    }

    float3 sample(inout RandomGenerator rng) {
        return vmf_sample_tokuyosh(vmf.xyz, vmf.w, rng.next_float2());
    }

    DistributionSample<float3> sample_eval(inout RandomGenerator rng) {
        const float3 s = sample(rng);
        return DistributionSample<float3>(s, pdf(s));
    }

    bool is_valid() {
        return vmf.w > 0.0;
    }

    float get_selection_probability(const float alpha) {
        return alpha < merian_ssmm_alpha_threshold ? 0.0 : merian_ssmm_probability;
    }
}

struct SSMMGuiding : GuidingModel {
    ParameterBlock<GBuffer> gbuffer;
    StructuredBuffer<SSMCState, ScalarDataLayout> prev;
    RWStructuredBuffer<SSMCState, ScalarDataLayout> next;

    struct State {
        // where the fit is written back; only the first vertex has one
        uint2 pixel;
        bool has_fit;
    }

    struct Record {
        SSMCState selected;
        bool guided;
    }

    typealias PathState = State;
    typealias DistributionImpl = SSMMDistribution;
    typealias UpdateRecord = Record;

    State begin_path() {
        State s;
        s.has_fit = false;
        return s;
    }

    SSMMDistribution get_distribution(inout State state,
                                      const RenderFrameInfo frame_info,
                                      const PathVertex vertex,
                                      inout RandomGenerator rng) {
        SSMMDistribution dist;
        dist.vmf = float4(0);
        dist.selected = SSMCState(float3(0), 0.0, 0, 0.0, 0.0);
        // a screen-space fit only describes the vertex the pixel actually sees
        if (vertex.bounce != uint16_t(1)) {
            return dist;
        }

        const uint2 resolution = gbuffer.get_dimensions();
        const float2 mv = gbuffer.get_motion_vectors(vertex.pixel);
        const int2 reprojected = int2(float2(vertex.pixel) + mv);

        // Resample the neighbourhood towards this vertex, walking outwards from the reprojected
        // pixel. Offsets are the sum of six uniforms, so the neighbourhood is sampled densely at
        // the centre and thinly at the edges.
        SSMCState tent = SSMCState(float3(0), 0.0, 0, 0.0, 0.0);
        float sum = 0.0;
        for (int i = 0; i < merian_ssmm_group_size; i++) {
            const float4 r0 = rng.next_float4();
            const float4 r1 = rng.next_float4();
            const float4 r2 = rng.next_float4();
            const int2 offset = int2(merian_ssmm_reuse_radius *
                                     (r0.xy + r0.zw + r1.xy + r1.zw + r2.xy + r2.zw - 3.0));
            const int2 tc = clamp(reprojected + offset, int2(0), int2(resolution) - 1);
            const SSMCState other = prev[ssmm_index(tc, resolution)];
            const float w = ssmm_reuse_weight(other, gbuffer, frame_info.cam_pos, resolution,
                                              vertex.pos, vertex.normal, tc);
            if (sum <= 0.0 || rng.next_float() < w / (w + sum)) {
                tent = other;
            }
            sum += w;
        }

        if (ssmm_valid(tent)) {
            dist.vmf = ssmm_vmf(tent, vertex.pos);
        }
        dist.selected = tent;

        state.pixel = vertex.pixel;
        state.has_fit = true;
        return dist;
    }

    Record record_scatter(const SSMMDistribution dist, const bool guided) {
        Record r;
        r.selected = dist.selected;
        r.guided = guided;
        return r;
    }

    float3 record_segment(inout State state,
                          const RenderFrameInfo frame_info,
                          const Record record,
                          const PathSegment segment,
                          inout RandomGenerator rng) {
        // a vMF lobe cannot represent a delta direction, and only the first vertex has a fit
        if (!state.has_fit || segment.delta || segment.bounce != uint16_t(1)) {
            return float3(0);
        }

        const float f = yuv_luminance(segment.incident * segment.weight);
        if (isinf(f) || isnan(f)) {
            return float3(0);
        }

        // a direction the fit did not propose explores: it starts a fit rather than continuing one
        SSMCState fit = record.guided ? record.selected : SSMCState(float3(0), 0.0, 0, 0.0, 0.0);
        fit.f = f;
        ssmm_add(fit, segment.pos, f, segment.wo, segment.next_pos);

        next[ssmm_index(int2(state.pixel), gbuffer.get_dimensions())] = fit;
        return float3(0);
    }
}

} // namespace merian
