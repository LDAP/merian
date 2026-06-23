#include "merian-shaders/light-cache/light-cache.slangh"

import merian_shaders.utils.grid;
import merian_shaders.utils.random;

namespace merian {

static const uint16_t HASHED_IRR_CACHE_MAX_N = 128;
static const float HASHED_IRR_CACHE_MIN_ALPHA = 0.01;

extern static const float lc_grid_tan_alpha_half = 0.002;
extern static const float lc_grid_steps_per_unit_size = 6.0;
extern static const float lc_grid_min_width = 0.001;
extern static const float lc_grid_power = 2.0;

float lc_target_grid_width(const float3 pos, const float3 cam_pos) {
    const float3 abs_dist = abs(cam_pos - pos);
    return 2 * lc_grid_tan_alpha_half * max(abs_dist.x, max(abs_dist.y, abs_dist.z));
}

uint lc_level_for_pos(const float3 pos, const float3 cam_pos) {
    const float w = lc_target_grid_width(pos, cam_pos);
    return uint(round(lc_grid_steps_per_unit_size *
                      log(max(w, lc_grid_min_width) / lc_grid_min_width) / log(lc_grid_power)));
}

float lc_grid_width_for_level(const uint level) {
    return lc_grid_min_width * pow(lc_grid_power, level / lc_grid_steps_per_unit_size);
}

struct HashedIrradianceCache {
    RWStructuredBuffer<HashedIrradianceCacheVertex, ScalarDataLayout> irr_cache;

    struct QueryResult {
        half3 irr;
        uint16_t N;
    };

    Optional<QueryResult>
    get_level(const uint level, const float3 pos, const float3 normal, inout RandomGenerator rng) {
        const int3 grid_idx =
            grid_idx_interpolate(pos, lc_grid_width_for_level(level), rng.next_float());
        const uint buf_idx =
            hash_grid_normal_level(uint3(grid_idx), normal, level, irr_cache.getCount());
        const HashedIrradianceCacheVertex vtx = irr_cache[buf_idx];

        if (vtx.hash == hash2_grid_level(grid_idx, level) && !any(isinf(vtx.irr)) &&
            !any(isnan(vtx.irr))) {
            return QueryResult(vtx.irr, vtx.N);
        }

        return none;
    }

    half3
    get(const float3 pos, const float3 normal, const float3 cam_pos, inout RandomGenerator rng) {
        const uint level = lc_level_for_pos(pos, cam_pos);
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
        const uint level = lc_level_for_pos(pos, cam_pos);
        if (let r = get_level(level, pos, normal, rng)) {
            return r.irr;
        } else if (any(cam_pos != prev_cam_pos)) {
            const uint prev_level = lc_level_for_pos(pos, prev_cam_pos);
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
        const uint level = lc_level_for_pos(pos, cam_pos);
        const int3 grid_idx =
            grid_idx_interpolate(pos, lc_grid_width_for_level(level), rng.next_float());
        const uint buf_idx =
            hash_grid_normal_level(uint3(grid_idx), normal, level, irr_cache.getCount());

        uint old;
        InterlockedExchange(irr_cache[buf_idx].lock, frame, old);
        if (old == frame) {
            // did not get lock
            return;
        }

        HashedIrradianceCacheVertex vtx = irr_cache[buf_idx];

        if (vtx.hash != hash2_grid_level(grid_idx, level) || any(isinf(vtx.irr)) ||
            any(isnan(vtx.irr))) {
            // Slot holds a different cell (or garbage): claim it for the current cell.
            vtx.irr = half3(0);
            vtx.N = uint16_t(0);
            vtx.hash = hash2_grid_level(grid_idx, level);

            // Seed from the previous-frame level to reconverge faster across camera motion.
            if (any(cam_pos != prev_cam_pos)) {
                let prev_level = lc_level_for_pos(pos, prev_cam_pos);
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

        irr_cache[buf_idx] = vtx;
        irr_cache[buf_idx].lock = 0;
    }
}
}
