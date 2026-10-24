#include "merian-shaders/utils/math-constants.slangh"

__exported import merian_shaders.scene.environment_map;

import merian_shaders.scene.light_data;
import merian_shaders.utils.envmap;
import merian_shaders.utils.morton;
import merian_shaders.utils.random;

namespace merian {

uint env_grid_shift(const uint size, const uint grid_log2) {
    const uint side_log2 = firstbithigh(size);
    return 2u * (side_log2 - min(side_log2, grid_log2));
}

// The environment luminance per texel of the equal-area octahedral square in Morton order, as
// leaves of about equal energy and their inclusive prefix sum. Every texel covers the same solid
// angle, so the density needs no jacobian.
struct EnvImportance {
    // per leaf its first texel, then the texel count
    StructuredBuffer<uint> keys;
    StructuredBuffer<float2> cdf;
    StructuredBuffer<uint> guide;
    // per texel of the lookup grid the leaf holding its first texel, then the leaf count
    StructuredBuffer<uint> lookup;
    // texels per side, 0 disables environment sampling
    uint size;
    uint buckets;

    uint texel_count() {
        return size * size;
    }

    uint leaf_count() {
        return size > 0 ? lookup[texel_count() >> env_grid_shift(size, ENV_LOOKUP_LOG2)] : 0u;
    }

    float get_total() {
        const float scale = size > 0 ? asfloat(guide[buckets + 1]) : 0.f;
        return scale > 0.f ? float(buckets) / scale : 0.f;
    }

    float get_average() {
        return size > 0 ? get_total() / float(texel_count()) : 0.f;
    }

    bool is_valid() {
        return get_total() > 0.f;
    }

    uint leaf_of(const uint texel) {
        const uint cell = texel >> env_grid_shift(size, ENV_LOOKUP_LOG2);
        uint lo = lookup[cell];
        uint hi = lookup[cell + 1];
        while (lo < hi) {
            const uint mid = (lo + hi + 1u) >> 1;
            if (keys[mid] <= texel)
                lo = mid;
            else
                hi = mid - 1u;
        }
        return lo;
    }

    float2 leaf_base(const uint leaf) {
        return leaf > 0 ? cdf[leaf - 1] : float2(0);
    }

    // the luminance summed over the texels before texel
    float2 prefix(const uint texel) {
        if (texel >= texel_count())
            return cdf[leaf_count() - 1];
        const uint leaf = leaf_of(texel);
        const float2 before = leaf_base(leaf);
        const uint first = keys[leaf];
        const float inside = float(texel - first) / float(keys[leaf + 1] - first);
        return ff_add(before, float2(ff_diff(cdf[leaf], before) * inside, 0.f));
    }

    float texel_value(const uint texel) {
        const uint leaf = leaf_of(texel);
        return max(ff_diff(cdf[leaf], leaf_base(leaf)), 0.f) / float(keys[leaf + 1] - keys[leaf]);
    }

    float range_share(const float2 prefix_first, const float2 prefix_end, const uint texel) {
        const float total = ff_diff(prefix_end, prefix_first);
        return total > 0.f ? texel_value(texel) / total : 0.f;
    }

    uint draw_texel(const uint first,
                    const uint end,
                    const float2 prefix_first,
                    const float2 prefix_end,
                    const float2 u) {
        const float2 threshold = range_threshold(prefix_first, prefix_end, u);
        const uint leaf = search_guided(cdf, guide, buckets, 0, leaf_count(), threshold);
        const float2 before = leaf_base(leaf);
        const float energy = ff_diff(cdf[leaf], before);
        const uint key = keys[leaf];
        const uint texels = keys[leaf + 1] - key;
        const float inside = energy > 0.f ? ff_diff(threshold, before) / energy : 0.f;
        return clamp(key + min(uint(max(inside, 0.f) * float(texels)), texels - 1u), first,
                     end - 1u);
    }

    // Returns the texel and fresh random numbers, which are uniform inside it.
    bool sample_texel(inout RandomGenerator rng, out uint texel, out float2 u) {
        texel = 0;
        u = rng.next_float2();
        if (!is_valid())
            return false;
        texel = draw_texel(0, texel_count(), float2(0), prefix(texel_count()), rng.next_float2());
        return true;
    }

    // A direction uniformly inside the texel, with the density of drawing the texel.
    Optional<EnvSample> texel_sample(const uint texel, const float2 u) {
        const float value = texel_value(texel);
        if (!(value > 0.f))
            return none;

        EnvSample s;
        s.dir = equal_area_square_to_sphere((float2(morton_decode2d(texel)) + u) / float(size));
        s.pdf = value / (get_average() * MERIAN_FOUR_PI);
        s.Le = float3(0);
        return s;
    }

    Optional<EnvSample> sample_direction(inout RandomGenerator rng) {
        uint texel;
        float2 u;
        if (!sample_texel(rng, texel, u))
            return none;
        return texel_sample(texel, u);
    }

    uint texel_of(const float3 dir) {
        return morton_encode2d(
            min(uint2(equal_area_sphere_to_square(dir) * float(size)), uint2(size - 1)));
    }

    float value(const float3 dir) {
        return texel_value(texel_of(dir));
    }

    // Solid angle density of sample_direction() generating dir.
    float pdf(const float3 dir) {
        if (!is_valid())
            return 0.f;
        return value(dir) / (get_average() * MERIAN_FOUR_PI);
    }
}

} // namespace merian
