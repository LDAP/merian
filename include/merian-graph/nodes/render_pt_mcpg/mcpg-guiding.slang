__exported import merian_shaders.sampling.guiding;
__exported import merian_shaders.light_cache.hashed_irradiance_cache;
__exported import mc;

import merian_shaders.colors.colorspaces;

using merian;

namespace merian {

extern static const int merian_guiding_mc_samples = 5;
extern static const float merian_guiding_probability = 0.9;
// scale the guiding probability with the lobe width
extern static const bool merian_guiding_scale_with_alpha = true;
// no guiding below this alpha: the BSDF is far more peaked than any lobe
extern static const float merian_guiding_alpha_threshold = 0.05;
extern static const bool merian_guiding_missing_light_heuristic = true;
// complete a truncated path with the cached irradiance at its last vertex
extern static const bool merian_guiding_light_cache_tail = false;
// floor on the density the irradiance cache divides by, bounding the variance it stores
extern static const float merian_guiding_lc_min_pdf = 1.0;

// The chain the continuation extends, and everything the update needs to write it back. Invalid
// where the direction did not come from the guiding: that starts a fresh chain.
struct MCPGRecord {
    MCState selected;
    HashGridPos selected_pos;
    // jittered normal the candidates were looked up with; the update must reuse it
    float3 access_normal;
    float score_sum;
}

// The Markov-chain candidates around a vertex. All of them are kept, not just the resampled one:
// their densities form the stochastic MIS weight.
struct MCPGDistribution : GuidingDistribution {
    VonMisesFisher3<false> vmfs[merian_guiding_mc_samples];
    float scores[merian_guiding_mc_samples];
    float score_sum;

    // the resampled chain, and the cell it was read from
    MCState selected;
    HashGridPos selected_pos;
    // jittered normal the candidates were looked up with
    float3 access_normal;

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

    float get_selection_probability(const float alpha) {
        if (alpha < merian_guiding_alpha_threshold) {
            return 0.0;
        }
        return merian_guiding_scale_with_alpha ? merian_guiding_probability * alpha
                                               : merian_guiding_probability;
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
    HashedIrradianceCache<LC_BUFFER_SIZE, LC_PROBE_COUNT, LC_STOCHASTIC_INTERPOLATION,
                          LC_SPLIT_STORAGE, LC_LOCALITY_BITS>
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
        const float3 lookup_pos = select(vertex.sample_index == uint16_t(0), vertex.prev_pos, vertex.pos);

        MCPGDistribution dist;
        dist.score_sum = 0.0;
        dist.selected = MCState();
        dist.selected_pos = HashGridPos();
        dist.access_normal = vertex.normal;

        for (int i = 0; i < merian_guiding_mc_samples; i++) {
            HashGridPos grid_pos;
            uint level;
            const MCState s = mcpg.load(lookup_pos, dist.access_normal, frame_info.cam_pos, frame_info.time,
                                        rng, grid_pos, level);
            dist.score_sum += s.sum_w;
            // comparison with NaN is false (when s.sum_w == 0 and score_sum == 0)
            if (rng.next_float() < s.sum_w / dist.score_sum) {
                dist.selected = s;
                dist.selected_pos = grid_pos;
                dist.vmfs[i] = dist.vmfs[0];
                dist.scores[i] = dist.scores[0];
                dist.scores[0] = s.sum_w;
                dist.vmfs[0] = s.get_vmf(vertex.pos);
            } else {
                dist.scores[i] = s.sum_w;
                dist.vmfs[i] = s.get_vmf(vertex.pos);
            }
        }

        return dist;
    }

    MCPGRecord record_scatter(const MCPGDistribution dist, const bool guided) {
        MCPGRecord record;
        // a direction the guiding did not propose starts a fresh chain, and must not invalidate
        // the cell the candidates came from
        record.selected = guided ? dist.selected : MCState();
        record.selected_pos = guided ? dist.selected_pos : HashGridPos();
        record.access_normal = dist.access_normal;
        record.score_sum = dist.score_sum;
        return record;
    }

    float3 record_segment(inout State state,
                      const RenderFrameInfo frame_info,
                      const MCPGRecord record,
                      const PathSegment segment,
                      inout RandomGenerator rng) {
        const HashGridPos update_pos = state.lc_grid_pos;
        state.lc_grid_pos = HashGridPos();

        // What the integrator will gather beyond this vertex: the emission where the continuation
        // found one, the cached irradiance where it is going to keep scattering. A path that stops
        // right here gathers nothing else, so it learns the emission alone.
        // a vMF lobe cannot represent a delta direction
        if (segment.delta) {
            return float3(0);
        }

        const bool emissive = any(segment.incident > 0.0);
        const bool gathers_emission_only =
            !merian_guiding_light_cache_tail && segment.is_last && segment.bounce == uint16_t(1);
        float3 target = segment.incident;
        if (segment.traced && !emissive && !gathers_emission_only) {
            target = float3(irr_cache.get(segment.next_pos, segment.next_normal, frame_info.cam_pos,
                                          frame_info.prev_cam_pos, rng, state.lc_grid_pos));
        }

        const float3 contribution = target * segment.weight;
        const float mc_f = yuv_luminance(contribution);
        if (isinf(mc_f) || isnan(mc_f)) {
            return float3(0);
        }

        // clamping the density bounds what a single near-specular sample can teach
        irr_cache.update(segment.pos, segment.normal,
                         target * segment.weight *
                             min(1.f, segment.pdf / merian_guiding_lc_min_pdf),
                         frame_info.cam_pos, frame_info.prev_cam_pos, frame_info.index, rng,
                         update_pos);

        if (rng.next_float() * record.score_sum < mc_f * merian_guiding_mc_samples) {
            MCState chain = record.selected;
            chain.add_sample(segment.pos, mc_f, segment.next_pos, segment.next_mv, frame_info.time);
            mcpg.store(chain, segment.pos, record.access_normal, frame_info.cam_pos, frame_info.index,
                       rng);
        } else if (merian_guiding_missing_light_heuristic && record.selected_pos.is_valid() &&
                   record.selected.light_missing_heuristic(mc_f, segment.wo, segment.pos,
                                                           segment.next_pos)) {
            mcpg.kill(record.selected_pos);
        }

        return merian_guiding_light_cache_tail && segment.traced && segment.is_last &&
                       !emissive
                   ? target
                   : float3(0);
    }
}

} // namespace merian
