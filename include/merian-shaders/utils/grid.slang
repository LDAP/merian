#pragma once

import "merian-shaders/utils/envmap.slang";

namespace merian {

// Determines the grid cell of pos and returns the lower vertex indices.
// The vertex pos then is index * cell_width.
int3 grid_idx_lower(const float3 pos, const float cell_width) {
    return int3(floor(pos / cell_width));
}

// Determines the grid cell of pos and returns the upper vertex indices.
// The vertex pos then is index * cell_width.
int3 grid_idx_upper(const float3 pos, const float cell_width) {
    return int3(ceil(pos / cell_width));
}

int2 grid_idx_lower(const float2 pos, const float cell_width) {
    return int2(floor(pos / cell_width));
}

int2 grid_idx_upper(const float2 pos, const float cell_width) {
    return int2(ceil(pos / cell_width));
}

// Determines the grid cell of pos and returns the closest vertex indices.
// The vertex pos then is index * cell_width.
int3 grid_idx_closest(const float3 pos, const float cell_width) {
    return int3(round(pos / cell_width));
}

// Returns a vertex index such that the expected value is equal to the interpolated value.
// Random must be uniformly drawn in [0, 1].
int3 grid_idx_interpolate(const float3 pos, const float cell_width, const float random) {
    const float3 grid_pos = fract(pos / cell_width);
    float bary_sum = 0;

    // (0, 0, 0)
    bary_sum += grid_pos.x * grid_pos.y * grid_pos.z;
    if (random <= bary_sum)
        return grid_idx_upper(pos, cell_width);

    bary_sum += grid_pos.x * grid_pos.y * (1. - grid_pos.z);
    if (random <= bary_sum)
        return select(bool3(0, 0, 1), grid_idx_lower(pos, cell_width),
                      grid_idx_upper(pos, cell_width));

    bary_sum += grid_pos.x * (1. - grid_pos.y) * grid_pos.z;
    if (random <= bary_sum)
        return select(bool3(0, 1, 0), grid_idx_lower(pos, cell_width),
                      grid_idx_upper(pos, cell_width));

    bary_sum += grid_pos.x * (1. - grid_pos.y) * (1. - grid_pos.z);
    if (random <= bary_sum)
        return select(bool3(0, 1, 1), grid_idx_lower(pos, cell_width),
                      grid_idx_upper(pos, cell_width));

    bary_sum += (1. - grid_pos.x) * grid_pos.y * grid_pos.z;
    if (random <= bary_sum)
        return select(bool3(1, 0, 0), grid_idx_lower(pos, cell_width),
                      grid_idx_upper(pos, cell_width));

    bary_sum += (1. - grid_pos.x) * grid_pos.y * (1. - grid_pos.z);
    if (random <= bary_sum)
        return select(bool3(1, 0, 1), grid_idx_lower(pos, cell_width),
                      grid_idx_upper(pos, cell_width));

    bary_sum += (1. - grid_pos.x) * (1. - grid_pos.y) * grid_pos.z;
    if (random <= bary_sum)
        return select(bool3(1, 1, 0), grid_idx_lower(pos, cell_width),
                      grid_idx_upper(pos, cell_width));

    // (1, 1, 1)
    return grid_idx_lower(pos, cell_width);
}

// Same as above for a 2d grid
int2 grid_idx_interpolate(const float2 pos, const float cell_width, const float random) {
    const float2 grid_pos = fract(pos / cell_width);
    float bary_sum = 0;

    // (0, 0)
    bary_sum += grid_pos.x * grid_pos.y;
    if (random <= bary_sum)
        return grid_idx_upper(pos, cell_width);

    bary_sum += grid_pos.x * (1. - grid_pos.y);
    if (random <= bary_sum)
        return select(bool2(0, 1), grid_idx_lower(pos, cell_width),
                      grid_idx_upper(pos, cell_width));

    bary_sum += (1. - grid_pos.x) * grid_pos.y;
    if (random <= bary_sum)
        return select(bool2(1, 0), grid_idx_lower(pos, cell_width),
                      grid_idx_upper(pos, cell_width));

    // (1, 1)
    return grid_idx_lower(pos, cell_width);
}

// See Müller et al. (2022): Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
// Müller et al. uses 1 for X but we observed artifacts when doing so
#define GRID_PRIME_X 1
#define GRID_PRIME_Y 2654435761
#define GRID_PRIME_Z 805459861

// Hashes the grid index, for x nothing is multiplied for better cache coherence.
uint hash_grid(const uint3 index, const uint modulus) {
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z)) %
           modulus;
}

// like hash_grid but with simple normal biasing
uint hash_grid_normal(const uint3 index, const float3 normal, const uint modulus) {
    const uint cube = cubemap_side(normal);
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z) ^
            (9351217 * cube + 13 * cube)) %
           modulus;
}

uint hash_grid_normal_level(const uint3 index,
                            const float3 normal,
                            const uint level,
                            const uint modulus) {
    const uint cube = cubemap_side(normal);
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z) ^
            (cube | (level << 16) * 723850877)) %
           modulus;
}

uint hash_grid_level(const uint3 index, const uint level, const uint modulus) {
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z) ^
            (723850877 * level + 231961 * level)) %
           modulus;
}

// Like hash_grid but modulus must be a power of two
uint hash_grid_2(const int3 index, const uint modulus_power_of_two) {
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z)) &
           (modulus_power_of_two - 1);
}

// Level in [0, max_level]. Higher levels have greater width.
// Claculates b such that min_width * b^max_level = max_width then min_width * b^level is returned.
float cell_width_for_level_geometric(const float level,
                                     const float max_level,
                                     const float min_width,
                                     const float max_width) {
    const float b = exp(log(max_width / min_width) / max_level);
    return min_width * pow(b, level);
}

// Level in [0, max_level]. Higher levels have greater width.
float cell_width_for_level_poly(const float level,
                                const float max_level,
                                const float min_width,
                                const float max_width,
                                const float degree) {
    return lerp(min_width, max_width, pow(level / max_level, degree));
}

// Level in [0, max_level]. Higher levels have greater width. (Special case of poly with degree 1)
float cell_width_for_level_linear(const float level,
                                  const float max_level,
                                  const float min_width,
                                  const float max_width) {
    return lerp(min_width, max_width, level / max_level);
}

#define GRID_PRIME_X_2 74763401
#define GRID_PRIME_Y_2 2254437599
#define GRID_PRIME_Z_2 508460413

// Second hash function for collision detection
uint hash2_grid(const int3 index) {
    return (index.x * GRID_PRIME_X_2) ^ (index.y * GRID_PRIME_Y_2) ^ (index.z * GRID_PRIME_Z_2);
}

// Second hash function for collision detection
uint hash2_grid_level(const int3 index, const uint level) {
    return (index.x * GRID_PRIME_X_2) ^ (index.y * GRID_PRIME_Y_2) ^ (index.z * GRID_PRIME_Z_2) ^
           (9351217 * level + 13 * level);
}

}
