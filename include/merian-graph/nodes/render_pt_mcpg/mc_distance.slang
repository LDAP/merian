import merian_shaders.utils.grid;
import merian_shaders.utils.random;

using merian;

static const uint MAX_DISTANCE_MC_VERTEX_STATE_COUNT = 10;

struct DistanceMCState {
    float sum_w;
    uint N;
    float2 moments;
};

struct DistanceMCVertex {
    DistanceMCState states[MAX_DISTANCE_MC_VERTEX_STATE_COUNT];
};

static const uint DISTANCE_ML_MAX_N = 1024;
static const float DISTANCE_ML_MIN_ALPHA = 0.01;

extern static const int distance_mc_grid_width = 25;
extern static const uint distance_mc_vertex_state_count = MAX_DISTANCE_MC_VERTEX_STATE_COUNT;

DistanceMCState distance_mc_state_new() {
    DistanceMCState r = {0.0, 0, float2(0)};
    return r;
}

// returns (mu, sigma)
float2 distance_mc_state_get_normal_dist(const DistanceMCState s) {
    const float2 moments = s.moments / (s.sum_w > 0.0 ? s.sum_w : 1.0);
    float sigma = sqrt(max(moments.y - moments.x * moments.x, 0.0));
    sigma = (s.N * s.N * sigma + 0.2) / (s.N * s.N + 0.2);
    return float2(moments.x, sigma);
}

// add sample to lobe via maximum likelihood estimator and exponentially weighted average
void distance_mc_state_add_sample(inout DistanceMCState s, const float distance, const float w) {
    s.N = min(s.N + 1, DISTANCE_ML_MAX_N);
    const float alpha = max(1.0 / float(s.N), DISTANCE_ML_MIN_ALPHA);

    s.sum_w = lerp(s.sum_w, w, alpha);
    s.moments = lerp(s.moments, w * float2(distance, distance * distance), alpha);
}

uint distance_mc_buf_idx(const int2 index, const uint grid_max_x) {
    return uint(index.x) + (grid_max_x + 1) * uint(index.y);
}

void distance_mc_load(RWStructuredBuffer<DistanceMCVertex> distance_mc_states,
                      out DistanceMCState s,
                      const float2 pixel,
                      const uint grid_max_x,
                      inout RandomGenerator rng) {
    const int2 grid_idx = grid_idx_interpolate(pixel, float(distance_mc_grid_width), rng.next_float());
    const uint state_idx = uint(rng.next_float() * distance_mc_vertex_state_count);

    s = distance_mc_states[distance_mc_buf_idx(grid_idx, grid_max_x)].states[state_idx];
}

void distance_mc_save(RWStructuredBuffer<DistanceMCVertex> distance_mc_states,
                      const DistanceMCState s,
                      const float2 pixel,
                      const uint grid_max_x,
                      inout RandomGenerator rng) {
    const int2 grid_idx = grid_idx_interpolate(pixel, float(distance_mc_grid_width), rng.next_float());
    const uint state_idx = uint(rng.next_float() * distance_mc_vertex_state_count);

    distance_mc_states[distance_mc_buf_idx(grid_idx, grid_max_x)].states[state_idx] = s;
}
