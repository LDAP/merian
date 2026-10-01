__exported import merian_shaders.sampling.guiding;
__exported import merian_shaders.light_cache.hashed_irradiance_cache;
__exported import merian_graph.nodes.guiding.mcpg.mc;

import merian_shaders.colors.colorspaces;
import merian_shaders.colors.debug;
import merian_shaders.utils.hash_grid;

using merian;

namespace merian {

extern static const int merian_guiding_mc_samples = 5;
extern static const float merian_guiding_weight_exponent = 1.0;
extern static const bool merian_guiding_missing_light_heuristic = true;
extern static const bool merian_guiding_light_cache_tail = false;
extern static const float merian_guiding_lc_min_pdf = 1.0;

// Invalid where the direction did not come from the guiding: that starts a fresh chain.
enum MCPGDebugView {
    Cell,
    Level,
    ChainWeight,
    Aim,
    LobeWidth,
    Samples,
    TargetMotion,
    Unresolved,
};

struct MCPGRecord {
    MCState selected;
    HashGridPos selected_pos;
    float score_sum;
}

// All candidates are kept: their densities form the stochastic MIS weight.
struct MCPGDistribution : GuidingDistribution<float3> {
    VonMisesFisher3<false> vmfs[merian_guiding_mc_samples];
    float scores[merian_guiding_mc_samples];
    float score_sum;
    float raw_sum;

    MCState selected;
    HashGridPos selected_pos;

    float pdf(const float3 s) {
        float p = 0.0;
        for (int i = 0; i < merian_guiding_mc_samples; i++) {
            p += scores[i] * vmfs[i].pdf(s);
        }
        return p / score_sum;
    }

    float3 sample(inout RandomGenerator rng) {
        return vmfs[0].sample(rng);
    }

    DistributionSample<float3> sample_eval(inout RandomGenerator rng) {
        const float3 s = sample(rng);
        return DistributionSample<float3>(s, pdf(s));
    }

    bool is_valid() {
        return score_sum > 0.0;
    }

    float get_confidence() {
        return 1.f;
    }
}

struct MCPGGuiding<let MC_BUFFER_SIZE : uint,
                   let MC_PROBE_COUNT : uint,
                   let MC_SPLIT_STORAGE : bool,
                   let MC_LOCALITY_BITS : uint,
                   let LC_BUFFER_SIZE : uint,
                   let LC_PROBE_COUNT : uint,
                   let LC_STOCHASTIC_INTERPOLATION : bool,
                   let LC_SPLIT_STORAGE : bool,
                   let LC_LOCALITY_BITS : uint> : GuidingModel {

    MCPG<MC_BUFFER_SIZE, MC_PROBE_COUNT, MC_SPLIT_STORAGE, MC_LOCALITY_BITS> mcpg;
    HashedIrradianceCache<LC_BUFFER_SIZE,
                          LC_PROBE_COUNT,
                          LC_STOCHASTIC_INTERPOLATION,
                          LC_SPLIT_STORAGE,
                          LC_LOCALITY_BITS>
        irr_cache;
    MCPGDebugView debug_selector;

    // where the irradiance cache was probed for this vertex, one bounce earlier
    struct State {
        HashGridPos lc_grid_pos;
    }

    typealias PathState = State;
    typealias DistributionImpl = MCPGDistribution;
    typealias UpdateRecord = MCPGRecord;

    State begin_path() {
        return {HashGridPos()};
    }

    MCPGDistribution get_distribution(inout State state,
                                      const RenderFrameInfo frame_info,
                                      const PathVertex vertex,
                                      inout RandomGenerator rng) {
        // the first sample finds last frame's lobe despite geometry motion
        const float3 lookup_pos =
            select(vertex.sample_index == uint16_t(0), vertex.prev_pos, vertex.pos);

        MCPGDistribution dist;
        dist.score_sum = 0.0;
        dist.raw_sum = 0.0;
        dist.selected = MCState();
        dist.selected_pos = HashGridPos();

        const float3 xi = float3(rng.next_float(), rng.next_float(), rng.next_float());

        for (int i = 0; i < merian_guiding_mc_samples; i++) {
            HashGridPos grid_pos;
            uint level;
            const MCState s =
                mcpg.load(lookup_pos, vertex.normal, frame_info.cam_pos, frame_info.time, uint(i),
                          uint(merian_guiding_mc_samples), xi, grid_pos, level);
            const float score = merian_guiding_weight_exponent == 1.f
                                    ? s.sum_w
                                    : pow(s.sum_w, merian_guiding_weight_exponent);
            dist.score_sum += score;
            dist.raw_sum += s.sum_w;
            // comparison with NaN is false (when s.sum_w == 0 and score_sum == 0)
            if (rng.next_float() < score / dist.score_sum) {
                dist.selected = s;
                dist.selected_pos = grid_pos;
                dist.vmfs[i] = dist.vmfs[0];
                dist.scores[i] = dist.scores[0];
                dist.scores[0] = score;
                dist.vmfs[0] = s.get_vmf(vertex.pos);
            } else {
                dist.scores[i] = score;
                dist.vmfs[i] = s.get_vmf(vertex.pos);
            }
        }

        return dist;
    }

    MCPGRecord record_scatter(const MCPGDistribution dist,
                              const PathSegment segment,
                              const bool guided,
                              inout RandomGenerator rng) {
        MCPGRecord record;
        // a direction the guiding did not propose must not invalidate the candidates' cell
        record.selected = guided ? dist.selected : MCState();
        record.selected_pos = guided ? dist.selected_pos : HashGridPos();
        record.score_sum = dist.raw_sum;
        return record;
    }

    float4 debug_view(const RenderFrameInfo frame_info,
                      const PathVertex vertex,
                      inout RandomGenerator rng) {
        const float3 cam = frame_info.cam_pos;
        if (debug_selector == MCPGDebugView::Cell) {
            const uint level = mcpg.grid.distributed_level_for_pos(vertex.pos, cam, rng);
            const int3 gidx =
                mcpg.grid.grid_idx(vertex.pos, mcpg.grid.grid_inv_width_for_level(level), rng);
            return float4(random_color(mcpg.grid.buffer_index(gidx, level)), 1);
        }
        if (debug_selector == MCPGDebugView::Level) {
            return float4(
                debug_color(mcpg.grid.distributed_level_for_pos(vertex.pos, cam, rng) / 63.0), 1);
        }

        const MCState s = mcpg.load(vertex.pos, vertex.shading_frame, cam, frame_info.time, rng);
        if (debug_selector == MCPGDebugView::Unresolved) {
            return float4(s.is_valid() ? 0.0 : 1.0);
        }
        if (!s.is_valid()) {
            return float4(1, 0, 1, 1);
        }
        switch (debug_selector) {
        case MCPGDebugView::ChainWeight:
            return float4(debug_color(s.sum_w * 0.1), 1);
        case MCPGDebugView::Aim:
            return float4((s.get_dir(vertex.pos) + 1) * 0.5, 1);
        case MCPGDebugView::LobeWidth:
            return float4(debug_color(s.one_minus_mean_cos(vertex.pos)), 1);
        case MCPGDebugView::Samples:
            return float4(debug_color(float(s.N) / float(ML_MAX_N)), 1);
        case MCPGDebugView::TargetMotion:
            return float4(abs(float3(s.mv)), 1);
        default:
            return float4(0);
        }
    }

    float3 record_segment(inout State state,
                          const RenderFrameInfo frame_info,
                          const MCPGRecord record,
                          const PathSegment segment,
                          inout RandomGenerator rng) {
        const HashGridPos update_pos = state.lc_grid_pos;
        state.lc_grid_pos = HashGridPos();

        if (segment.delta) {
            return float3(0);
        }

        const bool emissive = any(segment.incident > 0.0);
        const bool gathers_emission_only =
            !merian_guiding_light_cache_tail && segment.is_last && segment.bounce == uint16_t(1);

        // The emission where the continuation found one, the cached irradiance where it keeps
        // scattering. The lookup also resolves the cell the next vertex updates into.
        float3 target = segment.incident;
        if (segment.traced && !gathers_emission_only) {
            const float3 cached =
                float3(irr_cache.get(segment.next_pos, segment.next_normal, frame_info.cam_pos,
                                     frame_info.prev_cam_pos, rng, state.lc_grid_pos));
            if (!emissive) {
                target = cached;
            }
        }

        const float3 contribution = target * segment.weight;
        const float mc_f = yuv_luminance(contribution);
        if (isinf(mc_f) || isnan(mc_f)) {
            return float3(0);
        }

        // the density floor bounds what a single near-specular sample can teach
        irr_cache.update(segment.pos, segment.normal,
                         contribution * min(1.f, segment.pdf / merian_guiding_lc_min_pdf),
                         frame_info.cam_pos, frame_info.prev_cam_pos, frame_info.index, rng,
                         update_pos);

        const float3 tail =
            merian_guiding_light_cache_tail && segment.traced && segment.is_last && !emissive
                ? target
                : float3(0);

        // the cache keeps the emitter whole, the chains learn what the direct light policy allows
        const float direct_scale = emissive ? segment.direct_scale : 1.f;
        if (!(direct_scale > 0.f)) {
            return tail;
        }
        const float guide_f = mc_f * direct_scale;

        if (rng.next_float() * record.score_sum < guide_f * merian_guiding_mc_samples) {
            MCState chain = record.selected;
            chain.add_sample(segment.pos, guide_f, segment.aim_pos, segment.next_mv,
                             frame_info.time);
            mcpg.store(chain, segment.pos, segment.normal, frame_info.cam_pos, frame_info.index,
                       rng);
        } else if (merian_guiding_missing_light_heuristic && record.selected_pos.is_valid() &&
                   record.selected.light_missing_heuristic(guide_f, segment.wo, segment.pos,
                                                           segment.next_pos)) {
            mcpg.kill(record.selected_pos);
        }

        return tail;
    }
}

} // namespace merian
