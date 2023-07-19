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

#define GRID_PRIME_1 2654435761
#define GRID_PRIME_2 805459861

// Hashes the grid index, for x nothing is multiplied for better cache coherence.
// See Müller et al. (2022): Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
// and Lehmer (1951): Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.
uint hash_grid(const ivec3 index, const uint modulus) {
    return ((index.x ^ (index.y * GRID_PRIME_1) ^ (index.z * GRID_PRIME_2)) % modulus + modulus) % modulus;
}

// Like hash_grid but modulus must be a power of two
uint hash_grid_2(const ivec3 index, const uint modulus_power_of_two) {
    return (index.x ^ (index.y * GRID_PRIME_1) ^ (index.z * GRID_PRIME_2)) & (modulus_power_of_two - 1);
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

#endif
