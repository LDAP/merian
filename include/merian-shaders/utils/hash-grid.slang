import merian_shaders.utils.envmap;
import merian_shaders.utils.grid;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;

namespace merian {

static const uint MERIAN_HASH_GRID_LOCK_BIT = 0x80000000u;

struct HashGridVertex<typename T> {
    uint32_t hash;
    uint32_t lock;
    T data;
};

interface HashGridUpdateFunction<typename T> {
    void update(inout T t);
}

struct HashGrid<typename T,
                let BUFFER_SIZE : uint,
                let PROBE_COUNT : uint = 4,
                let STOCHASTIC_INTERPOLATION : bool = false> {
    RWStructuredBuffer<HashGridVertex<T>, ScalarDataLayout> buffer;

    float grid_tan_alpha_half;
    float grid_min_width;
    float grid_steps_per_unit_size;
    float grid_power;

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
        return xxhash32(uint4(gidx, level << 16 | cubemap_side(normal + 1e-3)));
    }

    uint buffer_index(const int3 gidx, const float3 normal, const uint level) {
        return hash_key(gidx, normal, level) % BUFFER_SIZE;
    }

    uint collision_key(const int3 gidx, const float3 normal, const uint level) {
        const uint cube = cubemap_side(normal + 1e-3);
        return xorshift128(uint4(uint4(gidx, level << 16 | cube)));
        // return (uint(gidx.x) & 0xFFu) | ((uint(gidx.y) & 0xFFu) << 8u) |
        //        ((uint(gidx.z) & 0xFFu) << 16u) | ((level & 0x1Fu) << 24u) | ((cube & 0x7u) <<
        //        29u);
    }

    Optional<T>
    get_level(const uint level, const float3 pos, const float3 normal, inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        uint buffer_idx = buffer_index(gidx, normal, level);
        const uint hash = collision_key(gidx, normal, level);

        for (uint i = 0; i < PROBE_COUNT; i++) {
            let vtx = buffer[buffer_idx];

            if (vtx.hash == hash) {
                return vtx.data;
            }

            buffer_idx++;
            if (buffer_idx >= BUFFER_SIZE)
                buffer_idx -= BUFFER_SIZE;
        }

        return none;
    }

    Optional<T>
    get(const float3 pos, const float3 normal, const float3 cam_pos, inout RandomGenerator rng) {
        const uint level = level_for_pos(pos, cam_pos, rng);
        return get_level(level, pos, normal, rng);
    }

    Optional<T> get(const float3 pos,
                    const float3 normal,
                    const float3 cam_pos,
                    const float3 prev_cam_pos,
                    inout RandomGenerator rng) {
        const uint level = level_for_pos(pos, cam_pos, rng);
        if (let r = get_level(level, pos, normal, rng)) {
            return r;
        } else if (any(cam_pos != prev_cam_pos)) {
            const uint prev_level = level_for_pos(pos, prev_cam_pos, rng);
            if (level != prev_level) {
                return get_level(prev_level, pos, normal, rng);
            }
        }

        return none;
    }

    void update(const float3 pos,
                const float3 normal,
                const float3 cam_pos,
                const float3 prev_cam_pos,
                const uint frame,
                HashGridUpdateFunction<T> update,
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
            let probe = buffer[idx];
            if (probe.hash == hash) {
                target = idx;
                break;
            }
            const uint age = frame - (probe.lock & ~MERIAN_HASH_GRID_LOCK_BIT);
            if (age >= max_age) {
                max_age = age;
                target = idx;
            }
        }

        // we use frame so we have it to evict the reast recently once and also to unlock all after
        // the frame implicitly.
        const uint locked = frame | MERIAN_HASH_GRID_LOCK_BIT;
        uint old;
        InterlockedExchange(buffer[target].lock, locked, old);
        if (old == locked) {
            return;
        }

        var vtx = buffer[target];

        if (vtx.hash != hash) {
            vtx.hash = hash;
            const uint prev_level = level_for_pos(pos, prev_cam_pos, rng);
            if (level != prev_level) {
                if (let r = get_level(prev_level, pos, normal, rng)) {
                    vtx.data = r;
                }
            }
        }

        update.update(vtx.data);

        vtx.lock = frame & ~MERIAN_HASH_GRID_LOCK_BIT;
        buffer[target] = vtx;
    }
}

} // namespace merian
