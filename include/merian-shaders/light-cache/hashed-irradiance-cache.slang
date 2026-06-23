#include "merian-shaders/light-cache/light-cache.slangh"

import merian_shaders.utils.grid;
import merian_shaders.utils.random;

namespace merian {

static const uint16_t HASHED_IRR_CACHE_MAX_N = 128;
static const float HASHED_IRR_CACHE_MIN_ALPHA = 0.01;

struct HashedIrradianceCache<let BUFFER_SIZE : uint,
                             let PROBE_COUNT : uint = 4,
                             let STOCHASTIC_INTERPOLATION : bool = false> {
    RWStructuredBuffer<HashedIrradianceCacheVertex, ScalarDataLayout> irr_cache;
    float grid_tan_alpha_half;
    float grid_min_width;
    float grid_steps_per_unit_size;
    float grid_power;

    struct QueryResult {
        half3 irr;
        uint16_t N;
    };

    float target_grid_width(const float3 pos, const float3 cam_pos) {
        const float3 abs_dist = abs(cam_pos - pos);
        return 2 * grid_tan_alpha_half * max(abs_dist.x, max(abs_dist.y, abs_dist.z));
    }

    uint level_for_pos(const float3 pos, const float3 cam_pos) {
        const float w = target_grid_width(pos, cam_pos);
        return uint(round(grid_steps_per_unit_size *
                          log(max(w, grid_min_width) / grid_min_width) / log(grid_power)));
    }

    float grid_width_for_level(const uint level) {
        return grid_min_width * pow(grid_power, level / grid_steps_per_unit_size);
    }

    int3 grid_idx(const float3 pos, const float width, inout RandomGenerator rng) {
        if (STOCHASTIC_INTERPOLATION)
            return grid_idx_interpolate(pos, width, rng.next_float());
        return grid_idx_closest(pos, width);
    }

    Optional<QueryResult>
    get_level(const uint level, const float3 pos, const float3 normal, inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        const uint base = hash_grid_normal_level(uint3(gidx), normal, level, BUFFER_SIZE);
        const uint hash = hash2_grid_level(gidx, level);

        for (uint i = 0; i < PROBE_COUNT; i++) {
            uint idx = base + i;
            if (idx >= BUFFER_SIZE)
                idx -= BUFFER_SIZE;
            const HashedIrradianceCacheVertex vtx = irr_cache[idx];
            if (vtx.hash == hash && !any(isinf(vtx.irr)) && !any(isnan(vtx.irr))) {
                return QueryResult(vtx.irr, vtx.N);
            }
        }

        return none;
    }

    half3
    get(const float3 pos, const float3 normal, const float3 cam_pos, inout RandomGenerator rng) {
        const uint level = level_for_pos(pos, cam_pos);
        if (let r = get_level(level, pos, normal, rng)) {
            return r.irr;
        }
        return half3(0);
    }

    half3 get(const float3 pos,
              const float3 normal,
              const float3 cam_pos,
              const float3 prev_cam_pos,
              inout RandomGenerator rng) {
        const uint level = level_for_pos(pos, cam_pos);
        if (let r = get_level(level, pos, normal, rng)) {
            return r.irr;
        } else if (any(cam_pos != prev_cam_pos)) {
            const uint prev_level = level_for_pos(pos, prev_cam_pos);
            if (level != prev_level) {
                if (let r = get_level(prev_level, pos, normal, rng)) {
                    return r.irr;
                }
            }
        }

        return half3(0);
    }

    void update(const float3 pos,
                const float3 normal,
                const float3 irr,
                const float3 cam_pos,
                const float3 prev_cam_pos,
                const uint frame,
                inout RandomGenerator rng) {
        const uint level = level_for_pos(pos, cam_pos);
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        const uint base = hash_grid_normal_level(uint3(gidx), normal, level, BUFFER_SIZE);
        const uint hash = hash2_grid_level(gidx, level);

        uint target = base;
        uint16_t min_N = uint16_t(0xFFFF);
        for (uint i = 0; i < PROBE_COUNT; i++) {
            uint idx = base + i;
            if (idx >= BUFFER_SIZE)
                idx -= BUFFER_SIZE;
            const HashedIrradianceCacheVertex probe = irr_cache[idx];
            if (probe.hash == hash) {
                target = idx;
                break;
            }
            if (probe.N < min_N) {
                min_N = probe.N;
                target = idx;
            }
        }

        uint old;
        InterlockedExchange(irr_cache[target].lock, frame, old);
        if (old == frame) {
            return;
        }

        HashedIrradianceCacheVertex vtx = irr_cache[target];

        if (vtx.hash != hash || any(isinf(vtx.irr)) || any(isnan(vtx.irr))) {
            vtx.irr = half3(0);
            vtx.N = uint16_t(0);
            vtx.hash = hash;

            if (any(cam_pos != prev_cam_pos)) {
                let prev_level = level_for_pos(pos, prev_cam_pos);
                if (prev_level != level) {
                    if (let result = get_level(prev_level, pos, normal, rng)) {
                        vtx.irr = result.irr;
                        vtx.N = uint16_t(1);
                    }
                }
            }
        }

        vtx.N = min(uint16_t(vtx.N + uint16_t(1)), HASHED_IRR_CACHE_MAX_N);
        vtx.irr =
            half3(lerp(float3(vtx.irr), irr, max(1.0 / float(vtx.N), HASHED_IRR_CACHE_MIN_ALPHA)));

        irr_cache[target] = vtx;
        irr_cache[target].lock = 0;
    }
}
}
