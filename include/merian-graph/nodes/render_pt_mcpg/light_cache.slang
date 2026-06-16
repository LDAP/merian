__exported import "grid.slang";
import merian_shaders.utils.grid;
import merian_shaders.utils.random;

using merian;

static const uint16_t LIGHT_CACHE_MAX_N = 128;
static const float LIGHT_CACHE_MIN_ALPHA = 0.01;

extern static const uint lc_grid_type = GRID_TYPE_EXPONENTIAL;
extern static const float lc_grid_tan_alpha_half = 0.002;
extern static const float lc_grid_steps_per_unit_size = 6.0;
extern static const float lc_grid_min_width = 0.01;
extern static const float lc_grid_power = 2.0;
extern static const uint light_cache_buffer_size = 4000000;

float lc_target_grid_width(const float3 pos, const float3 cam_pos) {
    return 2 * lc_grid_tan_alpha_half * distance(cam_pos, pos);
}

uint lc_level_for_pos(const float3 pos, const float3 cam_pos) {
    const float w = lc_target_grid_width(pos, cam_pos);
    if (lc_grid_type == GRID_TYPE_EXPONENTIAL) {
        return uint(round(lc_grid_steps_per_unit_size *
                          log(max(w, lc_grid_min_width) / lc_grid_min_width) / log(lc_grid_power)));
    } else {
        return uint(round(lc_grid_steps_per_unit_size *
                          pow(max(w - lc_grid_min_width, 0), 1 / lc_grid_power)));
    }
}

float lc_grid_width_for_level(const uint level) {
    if (lc_grid_type == GRID_TYPE_EXPONENTIAL) {
        return lc_grid_min_width * pow(lc_grid_power, level / lc_grid_steps_per_unit_size);
    } else {
        return pow(level / lc_grid_steps_per_unit_size, lc_grid_power) + lc_grid_min_width;
    }
}

void light_cache_get_level(RWStructuredBuffer<LightCacheVertex> light_cache,
                           out half3 irr,
                           out uint16_t N,
                           const uint level,
                           const float3 pos,
                           const float3 normal,
                           inout RandomGenerator rng) {
    const int3 grid_idx = grid_idx_interpolate(pos, lc_grid_width_for_level(level), rng.next_float());
    const uint buf_idx = hash_grid_normal_level(uint3(grid_idx), normal, level, light_cache_buffer_size);
    const LightCacheVertex vtx = light_cache[buf_idx];

    if (vtx.hash == hash2_grid_level(grid_idx, level) && !any(isinf(float3(vtx.irr))) &&
        !any(isnan(float3(vtx.irr)))) {
        irr = vtx.irr;
        N = vtx.N;
    } else {
        irr = half3(0);
        N = uint16_t(0);
    }
}

half3 light_cache_get(RWStructuredBuffer<LightCacheVertex> light_cache,
                      const float3 pos,
                      const float3 normal,
                      const float3 cam_pos,
                      inout RandomGenerator rng) {
    const uint level = lc_level_for_pos(pos, cam_pos);
    half3 irr;
    uint16_t N;
    light_cache_get_level(light_cache, irr, N, level, pos, normal, rng);
    return irr;
}

void light_cache_update(RWStructuredBuffer<LightCacheVertex> light_cache,
                        const float3 pos,
                        const float3 normal,
                        const float3 irr,
                        const float3 cam_pos,
                        const uint frame,
                        inout RandomGenerator rng) {
    const uint level = lc_level_for_pos(pos, cam_pos);
    const int3 grid_idx = grid_idx_interpolate(pos, lc_grid_width_for_level(level), rng.next_float());
    const uint buf_idx = hash_grid_normal_level(uint3(grid_idx), normal, level, light_cache_buffer_size);

    uint old;
    InterlockedExchange(light_cache[buf_idx].lock, frame, old);
    if (old == frame)
        // did not get lock
        return;

    LightCacheVertex vtx = light_cache[buf_idx];

    if (vtx.hash != hash2_grid_level(grid_idx, level) || any(isinf(float3(vtx.irr))) ||
        any(isnan(float3(vtx.irr)))) {

        // attempt to get from coarser level
        light_cache_get_level(light_cache, vtx.irr, vtx.N, level + 1, pos, normal, rng);
        vtx.hash = hash2_grid_level(grid_idx, level);
    }

    vtx.N = min(uint16_t(vtx.N + uint16_t(1)), LIGHT_CACHE_MAX_N);
    vtx.irr = half3(lerp(float3(vtx.irr), irr, max(1.0 / float(vtx.N), LIGHT_CACHE_MIN_ALPHA)));

    light_cache[buf_idx] = vtx;

    light_cache[buf_idx].lock = 0;
}
