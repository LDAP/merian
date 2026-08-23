__exported import merian_shaders.sampling.guiding;
import merian_shaders.utils.grid;
import merian_shaders.utils.textures;
import merian_shaders.utils.sampling;
import mc_distance;

using merian;

namespace merian {
// distance chains resampled towards this ray
extern static const int merian_distance_guiding_samples = 3;
// probability the volume pass gives the fitted distance over transmittance sampling
extern static const float merian_distance_guiding_probability = 0.9;
} // namespace merian

// Per-pixel Markov chains over the scattering distance, carried across frames by a forward
// projection into a screen-space grid.
namespace merian {

// The chains resampled around one ray, as a mixture of normal distributions. All candidates are
// kept: their densities form the stochastic MIS weight.
struct DistanceMCDistribution : GuidingDistribution<float> {
    float scores[merian_distance_guiding_samples];
    float2 normal_dists[merian_distance_guiding_samples];
    float score_sum;
    // the chain the update continues from, and the level it was read at
    DistanceMCState selected;
    uint selected_level;
    // the sampled distance is drawn while the candidates are still live
    float sampled;

    float pdf(const float s) {
        if (!(score_sum > 0.0)) {
            return 0.0;
        }
        float p = 0.0;
        for (int i = 0; i < merian_distance_guiding_samples; i++) {
            p += scores[i] * sample_normal_pdf(normal_dists[i].x, normal_dists[i].y, s);
        }
        return p / score_sum;
    }

    float sample(inout RandomGenerator rng) {
        return sampled;
    }

    DistributionSample<float> sample_eval(inout RandomGenerator rng) {
        const float s = sample(rng);
        return DistributionSample<float>(s, pdf(s));
    }

    bool is_valid() {
        return score_sum > 0.0;
    }

    float get_selection_probability(const float alpha) {
        return merian_distance_guiding_probability;
    }
}

struct MCPGDistanceGuiding : DistanceGuidingModel {
    DistanceMCGrid distance_mc;

    struct State {
        uint2 pixel;
        DistanceMCState chain;
    }

    struct Record {
        DistanceMCState selected;
        uint level;
        float score_sum;
        bool guided;
    }

    typealias PathState = State;
    typealias DistributionImpl = DistanceMCDistribution;
    typealias UpdateRecord = Record;

    State begin_path() {
        State s;
        s.chain = distance_mc_state_new();
        return s;
    }

    DistanceMCDistribution get_distribution(inout State state,
                                            const RenderFrameInfo frame_info,
                                            const RayVertex vertex,
                                            inout RandomGenerator rng) {
        DistanceMCDistribution dist;
        dist.score_sum = 0.0;
        dist.selected = distance_mc_state_new();
        dist.selected_level = 0;
        dist.sampled = 0.0;

        state.pixel = vertex.pixel;
        const float2 pixel = float2(vertex.pixel);

        for (int i = 0; i < merian_distance_guiding_samples; i++) {
            // the projection pass already moved the grid to the current camera
            const uint level = distance_mc.distributed_level(rng);
            const DistanceMCState s = distance_mc.load(pixel, level, rng);
            dist.normal_dists[i] = distance_mc_state_get_normal_dist(s);

            // a chain whose mean sits behind the medium describes light this ray cannot reach
            dist.scores[i] = s.sum_w * float(s.sum_w > 0.0) *
                             float(dist.normal_dists[i].x < vertex.max_distance);
            dist.score_sum += dist.scores[i];
            // comparison with NaN is false (when scores[i] == 0 and score_sum == 0)
            if (rng.next_float() < dist.scores[i] / dist.score_sum) {
                dist.selected = s;
                dist.selected_level = level;
                dist.sampled =
                    sample_normal_box_muller(dist.normal_dists[i].x, dist.normal_dists[i].y, rng).x;
            }
        }

        state.chain = dist.selected;
        return dist;
    }

    Record record_scatter(const DistanceMCDistribution dist, const bool guided) {
        Record r;
        r.selected = dist.selected;
        r.level = dist.selected_level;
        r.score_sum = dist.score_sum;
        r.guided = guided;
        return r;
    }

    void record_segment(inout State state,
                        const RenderFrameInfo frame_info,
                        const Record record,
                        const RaySegment segment,
                        inout RandomGenerator rng) {
        DistanceMCState chain = record.selected;
        distance_mc_state_add_sample(chain, segment.distance, segment.weight);
        if (rng.next_float() * record.score_sum <
            segment.weight * merian_distance_guiding_samples) {
            // sharp update at the level it was read from, plus a coarse propagation one level up
            distance_mc.store(chain, float2(state.pixel), record.level, rng);
            distance_mc.store(chain, float2(state.pixel),
                              min(record.level + 1, distance_mc_level_count - 1), rng);
        }
    }
}

} // namespace merian
