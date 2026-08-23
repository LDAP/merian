#include "merian-graph/nodes/render_ssmm/render_ssmm.slangh"

__exported import merian_shaders.sampling.guiding;
import merian_shaders.distributions.von_mises_fisher;
import merian_shaders.colors.colorspaces;
import merian_shaders.gbuffer;

using merian;

namespace merian {
// candidates resampled out of the screen neighbourhood, and mixed for the density
extern static const int merian_ssmm_candidates = 5;
// probability the integrator gives the fitted lobe
extern static const float merian_ssmm_probability = 0.85;
// no guiding below this lobe width: the BSDF is far more peaked than any fitted lobe
extern static const float merian_ssmm_alpha_threshold = 0.05;
// samples folded into the exponentially weighted fit
extern static const uint merian_ssmm_max_n = 1024u;
extern static const float merian_ssmm_min_alpha = 0.01;
extern static const float merian_ssmm_prior_n = 0.2;
}

// Screen-space mixture models (Dittebrandt et al. 2023), ported from VKDT with the author's
// approval. One fitted lobe per pixel, carried over frames through the motion vectors; it only
// describes the first scattering vertex, so deeper vertices get nothing.
namespace merian {

bool ssmm_valid(const SSMCState s) {
    return s.sum_w > 0.0;
}

float3 ssmm_target_dir(const SSMCState s, const float3 x) {
    return normalize(s.sum_tgt / (s.sum_w > 0.0 ? s.sum_w : 1.0) - x);
}

// vMF lobe (direction, concentration) the state stands for at x
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
                        const GBuffer gbuffer,
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

// The neighbourhood's fits around one vertex. All candidates are kept: their densities form the
// stochastic MIS weight, exactly as the Markov chain guiding does.
struct SSMMDistribution : GuidingDistribution {
    float4 vmfs[merian_ssmm_candidates];
    float scores[merian_ssmm_candidates];
    float score_sum;
    // the resampled fit the update continues from
    SSMCState selected;

    float pdf(const float3 s) {
        float p = 0.0;
        for (int i = 0; i < merian_ssmm_candidates; i++) {
            p += vmfs[i].w > 0.0 ? scores[i] * vmf_pdf_tokuyoshi(s, vmfs[i].xyz, vmfs[i].w) : 0.0;
        }
        return p / score_sum;
    }

    float3 sample(inout RandomGenerator rng) {
        return vmf_sample_tokuyosh(vmfs[0].xyz, vmfs[0].w, rng.next_float2());
    }

    DistributionSample<float3> sample_eval(inout RandomGenerator rng) {
        const float3 s = sample(rng);
        return DistributionSample<float3>(s, pdf(s));
    }

    bool is_valid() {
        return score_sum > 0.0 && vmfs[0].w > 0.0;
    }

    float get_selection_probability(const float alpha) {
        return alpha < merian_ssmm_alpha_threshold ? 0.0 : merian_ssmm_probability;
    }
}

struct SSMMGuiding : GuidingModel {
    SSMCStateAddress prev;
    SSMCStateAddress next;
    ParameterBlock<GBuffer> gbuffer;

    struct State {
        // where the fit is written back; only the first vertex has one
        uint2 pixel;
        bool has_fit;
        SSMCState fit;
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
        SSMMDistribution d;
        d.score_sum = 0.0;
        for (int i = 0; i < merian_ssmm_candidates; i++) {
            d.vmfs[i] = float4(0);
            d.scores[i] = 0.0;
        }
        // a screen-space fit only describes the vertex the pixel actually sees
        if (vertex.bounce != uint16_t(1)) {
            return d;
        }

        const uint2 resolution = gbuffer.get_dimensions();
        state.pixel = vertex.pixel;
        const float2 mv = gbuffer.get_motion_vectors(vertex.pixel);
        const int2 reprojected = int2(float2(vertex.pixel) + mv);

        // resample the neighbourhood towards this vertex; the walk is the same one the fit was
        // trained with, so the accepted candidate leads
        SSMCState tent =
            prev[ssmm_index(clamp(reprojected, int2(0), int2(resolution) - 1), resolution)];
        float sum = ssmm_reuse_weight(tent, gbuffer, frame_info.cam_pos, resolution, vertex.pos,
                                      vertex.normal, reprojected);
        for (int i = 0; i < merian_ssmm_candidates; i++) {
            const float2 r = rng.next_float2();
            const int2 offset = int2(30.0 * (r - 0.5));
            const int2 tc = clamp(reprojected + offset, int2(0), int2(resolution) - 1);
            const SSMCState other = prev[ssmm_index(tc, resolution)];
            const float w = ssmm_reuse_weight(other, gbuffer, frame_info.cam_pos, resolution,
                                              vertex.pos, vertex.normal, tc);
            if (sum <= 0.0 || rng.next_float() < w / (w + sum)) {
                tent = other;
            }
            sum += w;

            d.scores[i] = w;
            d.vmfs[i] = ssmm_valid(other) ? ssmm_vmf(other, vertex.pos) : float4(0);
            d.score_sum += w;
        }
        // the accepted fit leads, so that sample() draws from it
        d.selected = tent;
        if (ssmm_valid(tent)) {
            d.vmfs[0] = ssmm_vmf(tent, vertex.pos);
            d.scores[0] = max(d.scores[0], tent.f);
            d.score_sum = max(d.score_sum, d.scores[0]);
        }

        state.has_fit = true;
        state.fit = tent;
        return d;
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

        SSMCState fit = record.selected;
        const float f = yuv_luminance(segment.incident * segment.weight);
        if (!isinf(f) && !isnan(f)) {
            if (!record.guided) {
                // a BSDF sample explores: it starts a fit rather than continuing one
                fit = SSMCState(float3(0), 0.0, 0, 0.0, 0.0);
            }
            fit.f = f;
            ssmm_add(fit, segment.pos, f, segment.wo, segment.next_pos);
            state.fit = fit;
        }

        next[ssmm_index(int2(state.pixel), gbuffer.get_dimensions())] = state.fit;
        return float3(0);
    }
}

} // namespace merian
