#include "merian-shaders/light-cache/light-cache.slangh"

import merian_shaders.utils.envmap;
import merian_shaders.utils.grid;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;

namespace merian {

static const uint16_t HASHED_IRR_CACHE_MAX_N = 128;
static const float HASHED_IRR_CACHE_MIN_ALPHA = 0.01;
// High bit of the lock word marks an in-flight update; the low bits hold the last-touched frame.
static const uint HASHED_IRR_CACHE_LOCK_BIT = 0x80000000u;

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

    uint level_for_pos(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        const float w = target_grid_width(pos, cam_pos);
        const float level = grid_steps_per_unit_size *
                            log(max(w, grid_min_width) / grid_min_width) / log(grid_power);
        // if (STOCHASTIC_INTERPOLATION)
        //     return uint(floor(level) + select(rng.next_float() < fract(level), 1.0, 0.0));
        return uint(round(level));
    }

    float grid_width_for_level(const uint level) {
        return grid_min_width * pow(grid_power, level / grid_steps_per_unit_size);
    }

    int3 grid_idx(const float3 pos, const float width, inout RandomGenerator rng) {
        if (STOCHASTIC_INTERPOLATION)
            return grid_idx_interpolate(pos, width, rng.next_float());
        return grid_idx_closest(pos, width);
    }

    uint hash_key(const int3 gidx, const float3 normal, const uint level) {
        return xxhash32(uint4(gidx, level << 16 | cubemap_side(normal)));
    }

    uint buffer_index(const int3 gidx, const float3 normal, const uint level) {
        return hash_key(gidx, normal, level) % BUFFER_SIZE;
    }

    uint collision_key(const int3 gidx, const float3 normal, const uint level) {
        const uint cube = cubemap_side(normal);
        return xorshift128(uint4(uint4(gidx, level << 16 | cubemap_side(normal))));
        // return (uint(gidx.x) & 0xFFu) | ((uint(gidx.y) & 0xFFu) << 8u) |
        //        ((uint(gidx.z) & 0xFFu) << 16u) | ((level & 0x1Fu) << 24u) | ((cube & 0x7u) <<
        //        29u);
    }

    Optional<QueryResult>
    get_level(const uint level, const float3 pos, const float3 normal, inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        const uint base = buffer_index(gidx, normal, level);
        const uint hash = collision_key(gidx, normal, level);

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
        const uint level = level_for_pos(pos, cam_pos, rng);
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
        const uint level = level_for_pos(pos, cam_pos, rng);
        if (let r = get_level(level, pos, normal, rng)) {
            return r.irr;
        } else if (any(cam_pos != prev_cam_pos)) {
            const uint prev_level = level_for_pos(pos, prev_cam_pos, rng);
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
        const uint level = level_for_pos(pos, cam_pos, rng);
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        const uint base = buffer_index(gidx, normal, level);
        const uint hash = collision_key(gidx, normal, level);

        // Evict least-recently-touched slot.
        // nice: an uninitialized slot might have a timestamp in the future, so its unsigned age
        // wraps and is reclaimed first (self-healing).
        uint target = base;
        uint max_age = 0;
        for (uint i = 0; i < PROBE_COUNT; i++) {
            uint idx = base + i;
            if (idx >= BUFFER_SIZE)
                idx -= BUFFER_SIZE;
            const HashedIrradianceCacheVertex probe = irr_cache[idx];
            if (probe.hash == hash) {
                target = idx;
                break;
            }
            const uint age = frame - (probe.lock & ~HASHED_IRR_CACHE_LOCK_BIT);
            if (age >= max_age) {
                max_age = age;
                target = idx;
            }
        }

        const uint locked = frame | HASHED_IRR_CACHE_LOCK_BIT;
        uint old;
        InterlockedExchange(irr_cache[target].lock, locked, old);
        if (old == locked) {
            return;
        }

        HashedIrradianceCacheVertex vtx = irr_cache[target];

        if (vtx.hash != hash || any(isinf(vtx.irr)) || any(isnan(vtx.irr))) {
            vtx.irr = half3(0);
            vtx.N = uint16_t(0);
            vtx.hash = hash;

            if (any(cam_pos != prev_cam_pos)) {
                let prev_level = level_for_pos(pos, prev_cam_pos, rng);
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

        vtx.lock = frame & ~HASHED_IRR_CACHE_LOCK_BIT;
        irr_cache[target] = vtx;
    }
}
} // namespace merian
