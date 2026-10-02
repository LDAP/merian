#include "merian-shaders/utils/math-constants.slangh"

__exported import merian_shaders.scene.environment_map;

import merian_shaders.scene.light_data;
import merian_shaders.utils.envmap;
import merian_shaders.utils.morton;
import merian_shaders.utils.random;

namespace merian {

// The environment luminance per texel of the equal-area octahedral square, as an inclusive prefix
// sum in Morton order. Every texel covers the same solid angle, so the density needs no jacobian.
struct EnvImportance {
    StructuredBuffer<float2> cdf;
    // texels per side, 0 disables environment sampling
    uint size;

    uint texel_count() {
        return size * size;
    }

    float get_total() {
        if (size == 0)
            return 0.f;
        const float2 total = cdf[texel_count() - 1];
        return total.x + total.y;
    }

    float get_average() {
        return size > 0 ? get_total() / float(texel_count()) : 0.f;
    }

    bool is_valid() {
        return get_total() > 0.f;
    }

    // Returns the texel and fresh random numbers, which are uniform inside it.
    bool sample_texel(inout RandomGenerator rng, out uint2 texel, out float2 u) {
        texel = uint2(0);
        u = rng.next_float2();
        if (!is_valid())
            return false;
        texel = morton_decode2d(search_range(cdf, 0, texel_count(), rng.next_float2()));
        return true;
    }

    // A direction uniformly inside the texel, with the density of drawing the texel.
    Optional<EnvSample> texel_sample(const uint2 texel, const float2 u) {
        const float value = texel_value(texel);
        if (!(value > 0.f))
            return none;

        EnvSample s;
        s.dir = equal_area_square_to_sphere((float2(texel) + u) / float(size));
        s.pdf = value / (get_average() * MERIAN_FOUR_PI);
        s.Le = float3(0);
        return s;
    }

    Optional<EnvSample> sample_direction(inout RandomGenerator rng) {
        uint2 texel;
        float2 u;
        if (!sample_texel(rng, texel, u))
            return none;
        return texel_sample(texel, u);
    }

    // The pool holds texels drawn this frame by the same distribution.
    Optional<EnvSample> pool_sample(const StructuredBuffer<uint> pool,
                                    const uint pool_size,
                                    inout RandomGenerator rng) {
        if (!is_valid() || pool_size == 0)
            return none;
        const uint packed = pool[min(uint(rng.next_float() * pool_size), pool_size - 1)];
        if (packed == 0xFFFFFFFFu)
            return none;
        return texel_sample(uint2(packed & 0xFFFFu, packed >> 16u), rng.next_float2());
    }

    float texel_value(const uint2 texel) {
        const uint i = morton_encode2d(texel);
        return max(ff_diff(cdf[i], i > 0 ? cdf[i - 1] : float2(0)), 0.f);
    }

    uint2 texel_of(const float3 dir) {
        return min(uint2(equal_area_sphere_to_square(dir) * float(size)), uint2(size - 1));
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
