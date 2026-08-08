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

// Same as grid_idx_lower for a position already expressed in cell widths.
int3 grid_idx_lower_scaled(const float3 grid_pos) {
    return int3(floor(grid_pos));
}

// Returns a vertex index such that the expected value is equal to the interpolated value, for a
// position already expressed in cell widths. Random must be uniformly drawn in [0, 1].
int3 grid_idx_interpolate_scaled(const float3 grid_pos, const float random) {
    const int3 lower = int3(floor(grid_pos));
    const float3 f = grid_pos - float3(lower);
    float bary_sum = 0;

    bary_sum += f.x * f.y * f.z;
    if (random <= bary_sum)
        return lower + 1;

    bary_sum += f.x * f.y * (1. - f.z);
    if (random <= bary_sum)
        return lower + int3(1, 1, 0);

    bary_sum += f.x * (1. - f.y) * f.z;
    if (random <= bary_sum)
        return lower + int3(1, 0, 1);

    bary_sum += f.x * (1. - f.y) * (1. - f.z);
    if (random <= bary_sum)
        return lower + int3(1, 0, 0);

    bary_sum += (1. - f.x) * f.y * f.z;
    if (random <= bary_sum)
        return lower + int3(0, 1, 1);

    bary_sum += (1. - f.x) * f.y * (1. - f.z);
    if (random <= bary_sum)
        return lower + int3(0, 1, 0);

    bary_sum += (1. - f.x) * (1. - f.y) * f.z;
    if (random <= bary_sum)
        return lower + int3(0, 0, 1);

    return lower;
}

int3 grid_idx_interpolate(const float3 pos, const float cell_width, const float random) {
    return grid_idx_interpolate_scaled(pos / cell_width, random);
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

// -------------------------------------------------

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

}
