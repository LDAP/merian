import merian_shaders.utils.grid;
import merian_shaders.utils.textures;
import merian_shaders.utils.random;

using merian;

static const uint DISTANCE_ML_MAX_N = 1024;
static const float DISTANCE_ML_MIN_ALPHA = 0.01;

// Cell width in pixels at level 0; every further level doubles it, which is exactly one mip.
extern static const float distance_mc_base_width = 4.0;
// The host exports the exact count of views it binds.
extern static const uint distance_mc_level_count = 8;
// Effective dimensionality the levels are spread over. Keeps the expected number of threads per
// cell roughly constant; smaller widens the spread.
extern static const float distance_mc_distribution_dimension = 2.0;

struct DistanceMCState {
    float sum_w;
    uint N;
    float2 moments;

    // Packed one state per texel: N is small enough to survive the float round trip exactly.
    static DistanceMCState unpack(const float4 texel) {
        DistanceMCState s;
        s.sum_w = texel.x;
        s.N = uint(texel.y);
        s.moments = texel.zw;
        return s;
    }

    float4 pack() {
        return float4(sum_w, float(N), moments.x, moments.y);
    }
};

// One image per level, so a level is a plain mip lookup rather than a hash.
struct DistanceMCGrid {
    RWTexture2D<float4> levels[distance_mc_level_count];

    // Screen-space analogue of the world-space grid's distributed level: the projection already
    // keeps a cell's screen size constant, so only the geometric spread over levels remains.
    uint distributed_level(inout RandomGenerator rng) {
        const float level =
            -log2(1.0 - rng.next_float()) / distance_mc_distribution_dimension;
        return min(uint(round(level)), distance_mc_level_count - 1);
    }

    // The staggering pushes the index half a cell past the mip, so clamp rather than let the
    // border cells read zero and drop out of the resampling.
    uint2 cell(const float2 pixel, const uint level, inout RandomGenerator rng) {
        const float width = float(1u << level);
        // Offset by half a cell so levels of different width do not share vertices
        // (Mueller et al., "Instant Neural Graphics Primitives", Appendix A.)
        const float2 staggered = pixel / distance_mc_base_width + 0.5 * width;
        const int2 idx = grid_idx_interpolate(staggered, width, rng.next_float());
        return uint2(clamp(idx, int2(0), int2(texture_dimensions(levels[level])) - 1));
    }

    // Center of the pixel footprint a vertex represents; cell() without the jitter.
    float2 vertex_pixel(const uint2 idx, const uint level) {
        const float width = float(1u << level);
        return (float2(idx) - 0.5) * (distance_mc_base_width * width);
    }

    DistanceMCState load(const float2 pixel, const uint level, inout RandomGenerator rng) {
        return DistanceMCState::unpack(levels[level][cell(pixel, level, rng)]);
    }

    void store(const DistanceMCState s,
               const float2 pixel,
               const uint level,
               inout RandomGenerator rng) {
        levels[level][cell(pixel, level, rng)] = s.pack();
    }
}

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

// Shift every stored distance by delta (the ray origin moved along the view direction).
void distance_mc_state_shift(inout DistanceMCState s, const float delta) {
    s.moments = float2(s.moments.x + delta * s.sum_w,
                       s.moments.y + delta * (2.0 * s.moments.x + delta * s.sum_w));
}

// add sample to lobe via maximum likelihood estimator and exponentially weighted average
void distance_mc_state_add_sample(inout DistanceMCState s, const float distance, const float w) {
    s.N = min(s.N + 1, DISTANCE_ML_MAX_N);
    const float alpha = max(1.0 / float(s.N), DISTANCE_ML_MIN_ALPHA);

    s.sum_w = lerp(s.sum_w, w, alpha);
    s.moments = lerp(s.moments, w * float2(distance, distance * distance), alpha);
}
