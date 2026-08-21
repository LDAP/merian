#include "merian-shaders/utils/math-constants.slangh"

__exported import merian_shaders.scene.environment_map;

import merian_shaders.utils.envmap;
import merian_shaders.utils.morton;
import merian_shaders.utils.random;

namespace merian {

// A mip pyramid over the environment luminance in the equal-area octahedral square, sampled by
// hierarchical warping (Falcor's EnvMapSampler; McCool and Harwood 1997). Level 0 is the finest;
// every texel covers the same solid angle, so the density needs no jacobian.
struct EnvImportance {
    // levels concatenated finest first, each in Morton order: the four children of a texel are
    // contiguous, so one float4 load feeds a warp step
    StructuredBuffer<float4> levels;
    // 0 disables environment sampling
    uint size;
    uint level_count;

    // First quad of a level: the finer levels hold (size >> L)^2 / 4 quads each.
    uint level_offset(const uint level) {
        const uint quads = (size * size) / 4u;
        return (4u * (quads - (quads >> (2u * level)))) / 3u;
    }

    // The four children of the texel at the next coarser level.
    float4 load_quad(const uint level, const uint2 texel) {
        return levels[level_offset(level) + morton_encode2d(texel)];
    }

    // The coarsest quad holds the mean of every texel in its four entries.
    float get_average() {
        if (size == 0)
            return 0.f;
        const float4 top = levels[level_offset(level_count - 2)];
        return 0.25f * (top.x + top.y + top.z + top.w);
    }

    bool is_valid() {
        return size > 0 && get_average() > 0.f;
    }

    // Descends the pyramid, splitting the random numbers by the four children at every step.
    Optional<EnvSample> sample_direction(inout RandomGenerator rng) {
        if (!is_valid())
            return none;

        float2 u = rng.next_float2();
        uint2 texel = uint2(0);
        for (int level = int(level_count) - 2; level >= 0; level--) {
            // x = (0,0), y = (1,0), z = (0,1), w = (1,1) in Morton order
            const float4 w = load_quad(level, texel);
            texel <<= 1;

            // split vertically, then horizontally inside the chosen row
            const float top = w.x + w.y;
            const float total = top + w.z + w.w;
            if (!(total > 0.f))
                return none;
            const float p_top = top / total;
            const bool bottom = u.y >= p_top;
            u.y = bottom ? (u.y - p_top) / (1.f - p_top) : u.y / p_top;
            const float left = bottom ? w.z : w.x;
            const float row = bottom ? total - top : top;
            const float p_left = left / row;
            const bool right = u.x >= p_left;
            u.x = right ? (u.x - p_left) / (1.f - p_left) : u.x / p_left;
            texel += uint2(right ? 1 : 0, bottom ? 1 : 0);
        }

        const float value = texel_value(texel);
        if (!(value > 0.f))
            return none;

        EnvSample s;
        s.dir = equal_area_square_to_sphere((float2(texel) + u) / float(size));
        s.pdf = value / (get_average() * MERIAN_FOUR_PI);
        s.Le = float3(0);
        return s;
    }

    float texel_value(const uint2 texel) {
        const uint quad = morton_encode2d(texel >> 1);
        const uint child = (texel.x & 1u) | ((texel.y & 1u) << 1u);
        return levels[quad][child];
    }

    // Solid angle density of sample_direction() generating dir.
    float pdf(const float3 dir) {
        if (!is_valid())
            return 0.f;
        const float2 uv = equal_area_sphere_to_square(dir);
        const uint2 texel = min(uint2(uv * float(size)), uint2(size - 1));
        return texel_value(texel) / (get_average() * MERIAN_FOUR_PI);
    }
}

} // namespace merian
