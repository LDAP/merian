__exported import merian_shaders.sampling.guiding;
__exported import merian_shaders.light_cache.hashed_irradiance_cache;
__exported import mc;

import merian_shaders.colors.colorspaces;
import merian_shaders.colors.debug;

using merian;

namespace merian {

// What the chains make of an emitter the continuation found, where next event estimation covers
// the same light. In the mixture mode the density already carries the light technique, so the
// middle one is a no-op there and only separates the two in the resampled mode.

extern static const int merian_guiding_mc_samples = 5;
extern static const float merian_guiding_probability = 0.9;
// Exponent on a chain's weight where it selects among the candidates; the fixed point makes the
// probability proportional to the square root of the mass a chain represents.
extern static const float merian_guiding_weight_exponent = 1.0;
extern static const bool merian_guiding_missing_light_heuristic = true;
// The three direct illumination strategies of Rath et al., Variance-Aware Path Guiding
// (SIGGRAPH 2020). Their MIS compensation is a different weight again — the guiding's own share
// against the BSDF, on a squared target — and the chains do not carry it.
// complete a truncated path with the cached irradiance at its last vertex
extern static const bool merian_guiding_light_cache_tail = false;
// floor on the density the irradiance cache divides by, bounding the variance it stores
extern static const float merian_guiding_lc_min_pdf = 1.0;

// The chain the continuation extends, and everything the update needs to write it back. Invalid
// where the direction did not come from the guiding: that starts a fresh chain.
struct MCPGRecord {
    MCState selected;
    HashGridPos selected_pos;
    float score_sum;
}

// The Markov-chain candidates around a vertex. All of them are kept, not just the resampled one:
// their densities form the stochastic MIS weight.
struct MCPGDistribution : GuidingDistribution<float3> {
    VonMisesFisher3<false> vmfs[merian_guiding_mc_samples];
    float scores[merian_guiding_mc_samples];
    float score_sum;
    // sum of the raw chain weights, what a sample's own value is judged against
    float raw_sum;

    // the resampled chain, and the cell it was read from
    MCState selected;
    HashGridPos selected_pos;

    // Stochastic MIS over the candidates.
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

// Markov-chain path guiding over an adaptive hash grid, learning from an irradiance cache that
// stands in for the path tail.
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

    // Where the irradiance cache was already probed for this vertex, one bounce earlier.
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
        // temporally reproject the lookup for the first sample so the lobe written last frame is
        // found despite geometry motion
        const float3 lookup_pos =
            select(vertex.sample_index == uint16_t(0), vertex.prev_pos, vertex.pos);

        MCPGDistribution dist;
        dist.score_sum = 0.0;
        dist.raw_sum = 0.0;
        dist.selected = MCState();
        dist.selected_pos = HashGridPos();

        // one draw for the whole set: the level offset is stratified over it, the lattice vertex
        // again over the candidates that still share a level
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
        // a direction the guiding did not propose starts a fresh chain, and must not invalidate
        // the cell the candidates came from
        record.selected = guided ? dist.selected : MCState();
        record.selected_pos = guided ? dist.selected_pos : HashGridPos();
        record.score_sum = dist.raw_sum;
        return record;
    }

    // What the chains currently believe at this point; magenta where the grid holds nothing.
    bool debug_view(const int selector,
                    const RenderFrameInfo frame_info,
                    const PathVertex vertex,
                    inout RandomGenerator rng,
                    out float4 color) {
        color = float4(1, 0, 1, 1);
        if (selector <= 0) {
            return false;
        }

        const float3 cam = frame_info.cam_pos;
        if (selector == 1) { // the cell a lookup lands in
            const uint level = mcpg.grid.distributed_level_for_pos(vertex.pos, cam, rng);
            const int3 gidx =
                mcpg.grid.grid_idx(vertex.pos, mcpg.grid.grid_inv_width_for_level(level), rng);
            color = float4(random_color(mcpg.grid.buffer_index(gidx, level)), 1);
            return true;
        }
        if (selector == 2) { // its resolution level
            color = float4(
                debug_color(mcpg.grid.distributed_level_for_pos(vertex.pos, cam, rng) / 63.0), 1);
            return true;
        }

        const MCState s = mcpg.load(vertex.pos, vertex.shading_frame, cam, frame_info.time, rng);
        if (selector == 9) { // share of lookups that resolve no state at all
            color = float4(s.is_valid() ? 0.0 : 1.0);
            return true;
        }
        if (!s.is_valid()) {
            return true;
        }
        switch (selector) {
        case 3: // chain weight
            color = float4(debug_color(s.sum_w * 0.1), 1);
            return true;
        case 4: // aim
            color = float4((s.get_dir(vertex.pos) + 1) * 0.5, 1);
            return true;
        case 5: // lobe width, 1 - mean cos
            color = float4(debug_color(s.one_minus_mean_cos(vertex.pos)), 1);
            return true;
        case 6: // samples behind the fit
            color = float4(debug_color(float(s.N) / float(ML_MAX_N)), 1);
            return true;
        case 7: // motion of the target
            color = float4(abs(float3(s.mv)), 1);
            return true;
        }
        return false;
    }

    float3 record_segment(inout State state,
                          const RenderFrameInfo frame_info,
                          const MCPGRecord record,
                          const PathSegment segment,
                          inout RandomGenerator rng) {
        const HashGridPos update_pos = state.lc_grid_pos;
        state.lc_grid_pos = HashGridPos();

        // a vMF lobe cannot represent a delta direction
        if (segment.delta) {
            return float3(0);
        }

        const bool emissive = any(segment.incident > 0.0);
        const bool gathers_emission_only =
            !merian_guiding_light_cache_tail && segment.is_last && segment.bounce == uint16_t(1);

        // What the integrator will gather beyond this vertex: the emission where the continuation
        // found one, the cached irradiance where it is going to keep scattering. A path that stops
        // right here gathers nothing else, so it learns the emission alone. An emissive vertex
        // keeps its own radiance, but the lookup still runs: it resolves the cell the next vertex
        // updates into.
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

        // clamping the density bounds what a single near-specular sample can teach
        irr_cache.update(
            segment.pos, segment.normal,
            target * segment.weight * min(1.f, segment.pdf / merian_guiding_lc_min_pdf),
            frame_info.cam_pos, frame_info.prev_cam_pos, frame_info.index, rng, update_pos);

        // What the chains are worth steering towards. The cache above keeps the emitter whole
        // either way: it stands in for the whole path tail, not for the part the chains steer.
        const float direct_scale = emissive ? segment.direct_scale : 1.f;
        if (!(direct_scale > 0.f)) {
            // nothing to teach and nothing to hold against the cell either
            return merian_guiding_light_cache_tail && segment.traced && segment.is_last && !emissive
                       ? target
                       : float3(0);
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

        return merian_guiding_light_cache_tail && segment.traced && segment.is_last && !emissive
                   ? target
                   : float3(0);
    }
}

} // namespace merian
