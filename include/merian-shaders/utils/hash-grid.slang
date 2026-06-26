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

    float
    target_level_for_pos_f(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        const float w = target_grid_width(pos, cam_pos);
        return grid_steps_per_unit_size * log(max(w, grid_min_width) / grid_min_width) /
               log(grid_power);
    }

    uint target_level_for_pos(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        const float level = target_level_for_pos_f(pos, cam_pos, rng);
        // if (STOCHASTIC_INTERPOLATION)
        //     return uint(floor(level) + select(rng.next_float() < fract(level), 1.0, 0.0));
        return uint(round(level));
    }

    // Attemps to keep expected value of threads threads per cell constant
    uint
    distributed_level_for_pos(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        // / 2 if surface, / 3 if volume -> we use / 3 to bias a bit towards finer levels
        const float level =
            target_level_for_pos_f(pos, cam_pos, rng) +
            (-grid_steps_per_unit_size / 3 * log(1.0 - rng.next_float()) / log(grid_power));
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

    uint hash_key(const int3 gidx, const uint level) {
        return xxhash32(uint4(gidx, level));
    }

    uint buffer_index(const int3 gidx, const uint level) {
        return hash_key(gidx, level) % BUFFER_SIZE;
    }

    uint collision_key(const int3 gidx, const float3 normal, const uint level) {
        const uint cube = cubemap_side(normal + 1e-3);
        return xorshift128(uint4(uint4(gidx, level << 16 | cube)));
        // return (uint(gidx.x) & 0xFFu) | ((uint(gidx.y) & 0xFFu) << 8u) |
        //        ((uint(gidx.z) & 0xFFu) << 16u) | ((level & 0x1Fu) << 24u) | ((cube & 0x7u) <<
        //        29u);
    }

    uint collision_key(const int3 gidx, const uint level) {
        return xorshift128(uint4(uint4(gidx, level)));
    }

    Optional<T> get_grid_idx_at_level(const int3 gidx,
                                      const uint level,
                                      const uint hash,
                                      inout RandomGenerator rng) {
        uint buffer_idx = buffer_index(gidx, level);

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

    Optional<T> get_grid_idx_at_level(const int3 gidx,
                                      const uint level,
                                      const float3 normal,
                                      inout RandomGenerator rng) {
        const uint hash = collision_key(gidx, normal, level);
        return get_grid_idx_at_level(gidx, level, hash, rng);
    }

    Optional<T>
    get_grid_idx_at_level(const int3 gidx, const uint level, inout RandomGenerator rng) {
        const uint hash = collision_key(gidx, level);
        return get_grid_idx_at_level(gidx, level, hash, rng);
    }

    Optional<T>
    get_level(const uint level, const float3 pos, const float3 normal, inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        return get_grid_idx_at_level(gidx, level, normal, rng);
    }

    Optional<T>
    get(const float3 pos, const float3 normal, const float3 cam_pos, inout RandomGenerator rng) {
        const uint level = target_level_for_pos(pos, cam_pos, rng);
        return get_level(level, pos, normal, rng);
    }

    Optional<T> get_level(const uint level, const float3 pos, inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        return get_grid_idx_at_level(gidx, level, rng);
    }

    Optional<T> get(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        const uint level = target_level_for_pos(pos, cam_pos, rng);
        return get_level(level, pos, rng);
    }

    Optional<T> get(const float3 pos,
                    const float3 normal,
                    const float3 cam_pos,
                    const float3 prev_cam_pos,
                    inout RandomGenerator rng) {
        const uint level = target_level_for_pos(pos, cam_pos, rng);
        if (let r = get_level(level, pos, normal, rng)) {
            return r;
        } else if (any(cam_pos != prev_cam_pos)) {
            const uint prev_level = target_level_for_pos(pos, prev_cam_pos, rng);
            if (level != prev_level) {
                return get_level(prev_level, pos, normal, rng);
            }
        }

        return none;
    }

    Optional<T> distributed_get(const float3 pos,
                                const float3 normal,
                                const float3 cam_pos,
                                inout RandomGenerator rng) {
        const uint level = distributed_level_for_pos(pos, cam_pos, rng);
        return get_level(level, pos, normal, rng);
    }

    Optional<T> distributed_get(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        const uint level = distributed_level_for_pos(pos, cam_pos, rng);
        return get_level(level, pos, rng);
    }

    void update_at_level<let has_normal : bool>(const uint level,
                                                const float3 pos,
                                                const float3 normal,
                                                const float3 prev_cam_pos,
                                                const uint frame,
                                                HashGridUpdateFunction<T> update,
                                                inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        const uint base = buffer_index(gidx, level);

        uint hash;
        if (has_normal)
            hash = collision_key(gidx, normal, level);
        else
            hash = collision_key(gidx, level);

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
            const uint prev_level = target_level_for_pos(pos, prev_cam_pos, rng);
            if (level != prev_level) {
                Optional<T> r;
                if (has_normal)
                    r = get_level(prev_level, pos, normal, rng);
                else
                    r = get_level(prev_level, pos, rng);
                if (let v = r)
                    vtx.data = v;
            }
        }

        update.update(vtx.data);

        vtx.lock = frame & ~MERIAN_HASH_GRID_LOCK_BIT;
        buffer[target] = vtx;
    }

    void update(const float3 pos,
                const float3 normal,
                const float3 cam_pos,
                const float3 prev_cam_pos,
                const uint frame,
                HashGridUpdateFunction<T> update,
                inout RandomGenerator rng) {
        update_at_level<true>(target_level_for_pos(pos, cam_pos, rng), pos, normal, prev_cam_pos,
                              frame, update, rng);
    }

    void update(const float3 pos,
                const float3 cam_pos,
                const float3 prev_cam_pos,
                const uint frame,
                HashGridUpdateFunction<T> update,
                inout RandomGenerator rng) {
        update_at_level<false>(target_level_for_pos(pos, cam_pos, rng), pos,
                               float3(0), prev_cam_pos, frame, update, rng);
    }

    void distributed_update(const float3 pos,
                            const float3 normal,
                            const float3 cam_pos,
                            const float3 prev_cam_pos,
                            const uint frame,
                            HashGridUpdateFunction<T> update,
                            inout RandomGenerator rng) {
        update_at_level<true>(distributed_level_for_pos(pos, cam_pos, rng), pos, normal,
                              prev_cam_pos, frame, update, rng);
    }

    void distributed_update(const float3 pos,
                            const float3 cam_pos,
                            const float3 prev_cam_pos,
                            const uint frame,
                            HashGridUpdateFunction<T> update,
                            inout RandomGenerator rng) {
        update_at_level<false>(distributed_level_for_pos(pos, cam_pos, rng), pos,
                               float3(0), prev_cam_pos, frame, update, rng);
    }
}

} // namespace merian
