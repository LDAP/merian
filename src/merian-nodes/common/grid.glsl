#include "common/cubemap.glsl"

#ifndef _GRID_H_
#define _GRID_H_

// Determines the grid cell of pos and returns the lower vertex indices.
// The vertex pos then is index * cell_width.
ivec3 grid_idx_lower(const vec3 pos, const float cell_width) {
    return ivec3(floor(pos / cell_width));
}

// Determines the grid cell of pos and returns the upper vertex indices.
// The vertex pos then is index * cell_width.
ivec3 grid_idx_upper(const vec3 pos, const float cell_width) {
    return ivec3(ceil(pos / cell_width));
}

ivec2 grid_idx_lower(const vec2 pos, const float cell_width) {
    return ivec2(floor(pos / cell_width));
}

ivec2 grid_idx_upper(const vec2 pos, const float cell_width) {
    return ivec2(ceil(pos / cell_width));
}

// Determines the grid cell of pos and returns the closest vertex indices.
// The vertex pos then is index * cell_width.
ivec3 grid_idx_closest(const vec3 pos, const float cell_width) {
    return ivec3(round(pos / cell_width));
}

// Returns a vertex index such that the expected value is equal to the interpolated value.
// Random must be uniformly drawn in [0, 1].
ivec3 grid_idx_interpolate(const vec3 pos, const float cell_width, const float random) {
    const ivec3 lower = grid_idx_lower(pos, cell_width);
    const ivec3 upper = grid_idx_upper(pos, cell_width);
    const vec3 grid_pos = fract(pos / cell_width);

    float bary_sum = 0;
    
    // (0, 0, 0)
    bary_sum += grid_pos.x * grid_pos.y * grid_pos.z;
    if (random <= bary_sum)
        return upper;

    ivec3 vtx = ivec3(0, 0, 1);
    vec3 dist = abs(grid_pos - vtx);
    bary_sum += dist.x * dist.y * dist.z;
    if (random <= bary_sum)
        return vtx * lower + (1 - vtx) * upper;

    vtx = ivec3(0, 1, 0);
    dist = abs(grid_pos - vtx);
    bary_sum += dist.x * dist.y * dist.z;
    if (random <= bary_sum)
        return vtx * lower + (1 - vtx) * upper;

    vtx = ivec3(0, 1, 1);
    dist = abs(grid_pos - vtx);
    bary_sum += dist.x * dist.y * dist.z;
    if (random <= bary_sum)
        return vtx * lower + (1 - vtx) * upper;

    vtx = ivec3(1, 0, 0);
    dist = abs(grid_pos - vtx);
    bary_sum += dist.x * dist.y * dist.z;
    if (random <= bary_sum)
        return vtx * lower + (1 - vtx) * upper;

    vtx = ivec3(1, 0, 1);
    dist = abs(grid_pos - vtx);
    bary_sum += dist.x * dist.y * dist.z;
    if (random <= bary_sum)
        return vtx * lower + (1 - vtx) * upper;

    vtx = ivec3(1, 1, 0);
    dist = abs(grid_pos - vtx);
    bary_sum += dist.x * dist.y * dist.z;
    if (random <= bary_sum)
        return vtx * lower + (1 - vtx) * upper;

    // (1, 1, 1)
    return lower;
}

// Same as above for a 2d grid
ivec2 grid_idx_interpolate(const vec2 pos, const float cell_width, const float random) {
    const ivec2 lower = grid_idx_lower(pos, cell_width);
    const ivec2 upper = grid_idx_upper(pos, cell_width);
    const vec2 grid_pos = fract(pos / cell_width);

    float bary_sum = 0;

    // (0, 0)
    bary_sum += grid_pos.x * grid_pos.y;
    if (random <= bary_sum)
        return upper;

    ivec2 vtx = ivec2(0, 1);
    vec2 dist = abs(grid_pos - vtx);
    bary_sum += dist.x * dist.y;
    if (random <= bary_sum)
        return vtx * lower + (1 - vtx) * upper;

    vtx = ivec2(1, 0);
    dist = abs(grid_pos - vtx);
    bary_sum += dist.x * dist.y;
    if (random <= bary_sum)
        return vtx * lower + (1 - vtx) * upper;

    // (1, 1)
    return lower;
}

// See Müller et al. (2022): Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
// Müller et al. uses 1 for X but we observed artifacts when doing so
#define GRID_PRIME_X 44761931
#define GRID_PRIME_Y 2654435761
#define GRID_PRIME_Z 805459861

// Hashes the grid index, for x nothing is multiplied for better cache coherence.
uint hash_grid(const uvec3 index, const uint modulus) {
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z)) % modulus;
}

// like hash_grid but with simple normal biasing
uint hash_grid_normal(const uvec3 index, const vec3 normal, const uint modulus) {
    const uint cube = cubemap_side(normal);
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z) ^ (9351217 * cube + 13 * cube)) % modulus;
}

uint hash_grid_normal_level(const uvec3 index, const vec3 normal, const uint level, const uint modulus) {
    const uint cube = cubemap_side(normal);
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z) ^ (pack32(u16vec2(cube, level)) * 723850877)) % modulus;
}

uint hash_grid_level(const uvec3 index, const uint level, const uint modulus) {
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z) ^ (723850877 * level + 231961 * level)) % modulus;
}

// Like hash_grid but modulus must be a power of two
uint hash_grid_2(const ivec3 index, const uint modulus_power_of_two) {
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z)) & (modulus_power_of_two - 1);
}

// Level in [0, max_level]. Higher levels have greater width.
// Claculates b such that min_width * b^max_level = max_width then min_width * b^level is returned.
float cell_width_for_level_geometric(const float level, const float max_level, const float min_width, const float max_width) {
    const float b = exp(log(max_width / min_width) / max_level);
    return min_width * pow(b, level);
}

// Level in [0, max_level]. Higher levels have greater width.
float cell_width_for_level_poly(const float level, const float max_level, const float min_width, const float max_width, const float degree) {
    return mix(min_width, max_width, pow(level / max_level, degree));
}


// Level in [0, max_level]. Higher levels have greater width. (Special case of poly with degree 1)
float cell_width_for_level_linear(const float level, const float max_level, const float min_width, const float max_width) {
    return mix(min_width, max_width, level / max_level);
}

#define GRID_PRIME_X_2 74763401
#define GRID_PRIME_Y_2 2254437599
#define GRID_PRIME_Z_2 508460413

// Second hash function for collision detection
uint hash2_grid(const ivec3 index) {
    return (index.x * GRID_PRIME_X_2) ^ (index.y * GRID_PRIME_Y_2) ^ (index.z * GRID_PRIME_Z_2);
}

// Second hash function for collision detection
uint hash2_grid_level(const ivec3 index, const uint level) {
    return (index.x * GRID_PRIME_X_2) ^ (index.y * GRID_PRIME_Y_2) ^ (index.z * GRID_PRIME_Z_2) ^ (9351217 * level + 13 * level);
}

#endif
