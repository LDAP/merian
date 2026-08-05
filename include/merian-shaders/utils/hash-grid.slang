import merian_shaders.utils.encoding;
import merian_shaders.utils.envmap;
import merian_shaders.utils.grid;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;

namespace merian {

static const uint MERIAN_HASH_GRID_LOCK_BIT = 0x80000000u;
// Shifts the base level for a unit, allows resuolution up to ~2^-31 units
static const float MERIAN_HASH_GRID_LEVEL_OFFSET = 32.0;

interface HashGridUpdateFunction<typename T> {
    void update(inout T t);
}

struct HashGridSlot<typename T, let SPLIT_STORAGE : bool> {
    uint2 key[SPLIT_STORAGE ? 0 : 1];
    T t;
}

struct HashGrid<typename T,
                let BUFFER_SIZE : uint,
                let PROBE_COUNT : uint = 4,
                let STOCHASTIC_INTERPOLATION : bool = false,
                let LEVEL_STOCHASTIC_INTERPOLATION : bool = false,
                let SPLIT_STORAGE : bool = true,
                let GRID_POWER : uint = 2,
                let GRID_STEPS_PER_UNIT_SIZE : uint = 1> {

    // If SPLIT_STORAGE == true
    RWStructuredBuffer<uint2> keys;
    RWStructuredBuffer<HashGridSlot<T, SPLIT_STORAGE>, ScalarDataLayout> data;

    uint load_hash(const uint index) {
        if (SPLIT_STORAGE)
            return keys[index].x;
        return data[index].key[0].x;
    }

    uint2 load_key(const uint index) {
        if (SPLIT_STORAGE)
            return keys[index];
        return data[index].key[0];
    }

    void lock_exchange(const uint index, const uint locked, out uint old) {
        if (SPLIT_STORAGE)
            InterlockedExchange(keys[index].y, locked, old);
        else
            InterlockedExchange(data[index].key[0].y, locked, old);
    }

    // Payload before key so a concurrent probe never matches an unwritten payload.
    void store_slot(const uint index, const uint2 key, const T t) {
        data[index].t = t;
        if (SPLIT_STORAGE)
            keys[index] = key;
        else
            data[index].key[0] = key;
    }

    float grid_tan_alpha_half;
    // SHARC-style LOD bias: shifts the level quantization / near-camera detail (0 = neutral).
    float grid_level_bias;
    // Effective spatial dimensionality the distributed levels are spread over (2 = surface,
    // 3 = volume). Keeps the expected threads per cell roughly constant; smaller widens the spread.
    float grid_distribution_dimension;

    uint target_level_for_pos(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        const float3 d = cam_pos - pos;
        const float dist2 = max(dot(d, d), 1e-12);
        const float level = max(GRID_STEPS_PER_UNIT_SIZE * (0.5 * log(dist2) / log(GRID_POWER) +
                                                            MERIAN_HASH_GRID_LEVEL_OFFSET) +
                                    grid_level_bias,
                                1.0);
        if (LEVEL_STOCHASTIC_INTERPOLATION)
            return uint(floor(level) + select(rng.next_float() < fract(level), 1.0, 0.0));
        return uint(floor(level));
    }

    // Attemps to keep expected value of threads threads per cell constant
    uint
    distributed_level_for_pos(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        const float level = target_level_for_pos(pos, cam_pos, rng) +
                            (-float(GRID_STEPS_PER_UNIT_SIZE) / grid_distribution_dimension *
                             log(1.0 - rng.next_float()) / log(GRID_POWER));
        return uint(round(level));
    }

    float grid_width_for_level(const uint level) {
        return 2 * grid_tan_alpha_half *
               pow(GRID_POWER, (float(level) - grid_level_bias) / float(GRID_STEPS_PER_UNIT_SIZE) -
                                   MERIAN_HASH_GRID_LEVEL_OFFSET);
    }

    int3 grid_idx(const float3 pos, const float width, inout RandomGenerator rng) {
        // Offset the lattice by half a cell so that levels of different width don't share vertices
        // (Müller et al., "Instant Neural Graphics Primitives", Appendix A.)
        const float3 staggered = pos + 0.5 * width;
        if (STOCHASTIC_INTERPOLATION)
            return grid_idx_interpolate(staggered, width, rng.next_float());
        return grid_idx_lower(staggered, width);
    }

    // Dominant-axis cube face of the normal (6 bins).
    uint normal_bin(const float3 normal) {
        return cubemap_side(normal);
    }

    // Pack the LOD level and the cube face into one word (face in the low 3 bits).
    uint level_normal_word(const uint level, const float3 normal) {
        return (level << 3) | normal_bin(normal);
    }

    uint level_normal_word(const uint level) {
        return level << 3;
    }

    // Stored/compared key: a hash of (gidx, level, normal).
    uint collision_key(const int3 gidx, const float3 normal, const uint level) {
        return pcg4d16(uint4(gidx, level_normal_word(level, normal)));
    }

    uint collision_key(const int3 gidx, const uint level) {
        return pcg4d16(uint4(gidx, level_normal_word(level)));
    }

    // Slot from the key, so the hashed-in normal also selects the bucket.
    // Makes blocks of PROBE_COUNT slots compute the same starting index into the buffer for coalescing memory accesses.
    uint buffer_index(const uint key) {
        const uint slot = key % BUFFER_SIZE;
        if ((PROBE_COUNT & (PROBE_COUNT - 1u)) != 0)
            return slot;
        return slot & ~(PROBE_COUNT - 1u);
    }

    // Spatial (normal-less) slot, for debug coloring by cell.
    uint buffer_index(const int3 gidx, const uint level) {
        return buffer_index(collision_key(gidx, level));
    }

    bool keys_match(const uint stored, const uint query) {
        return stored == query;
    }

    // Linear-probe PROBE_COUNT slots starting at buffer_index for a matching hash.
    Optional<T> get(uint buffer_index, const uint hash) {
        for (uint i = 0; i < PROBE_COUNT; i++) {
            if (keys_match(load_hash(buffer_index), hash)) {
                return data[buffer_index].t;
            }

            buffer_index++;
            if (buffer_index >= BUFFER_SIZE)
                buffer_index -= BUFFER_SIZE;
        }

        return none;
    }

    // Linear-probe variant that also reports the matched absolute slot (0xFFFFFFFF if none).
    Optional<T> get(uint buffer_index, const uint hash, out uint matched_index) {
        for (uint i = 0; i < PROBE_COUNT; i++) {
            if (keys_match(load_hash(buffer_index), hash)) {
                matched_index = buffer_index;
                return data[buffer_index].t;
            }

            buffer_index++;
            if (buffer_index >= BUFFER_SIZE)
                buffer_index -= BUFFER_SIZE;
        }

        matched_index = 0xFFFFFFFFu;
        return none;
    }

    // Debug only: octahedral bin of the normal if an exact slot matches at the target level, else
    // 0xFFFFFFFF.
    uint debug_selected_bin(const float3 pos,
                            const float3 normal,
                            const float3 cam_pos,
                            inout RandomGenerator rng) {
        const uint level = target_level_for_pos(pos, cam_pos, rng);
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        const uint hash = collision_key(gidx, normal, level);
        uint bi = buffer_index(hash);
        for (uint i = 0; i < PROBE_COUNT; i++) {
            if (keys_match(load_hash(bi), hash))
                return normal_bin(normal);
            bi++;
            if (bi >= BUFFER_SIZE)
                bi -= BUFFER_SIZE;
        }
        return 0xFFFFFFFFu;
    }

    Optional<T> get_grid_idx_at_level(const int3 gidx,
                                      const uint level,
                                      const uint hash,
                                      inout RandomGenerator rng) {
        return get(buffer_index(hash), hash);
    }

    Optional<T> get_grid_idx_at_level(const int3 gidx,
                                      const uint level,
                                      const uint hash,
                                      inout RandomGenerator rng,
                                      out uint matched_index) {
        return get(buffer_index(hash), hash, matched_index);
    }

    Optional<T> get_grid_idx_at_level(const int3 gidx,
                                      const uint level,
                                      const float3 normal,
                                      inout RandomGenerator rng) {
        const uint hash = collision_key(gidx, normal, level);
        return get_grid_idx_at_level(gidx, level, hash, rng);
    }

    Optional<T> get_grid_idx_at_level(const int3 gidx,
                                      const uint level,
                                      const float3 normal,
                                      inout RandomGenerator rng,
                                      out uint matched_index) {
        const uint hash = collision_key(gidx, normal, level);
        return get_grid_idx_at_level(gidx, level, hash, rng, matched_index);
    }

    Optional<T>
    get_grid_idx_at_level(const int3 gidx, const uint level, inout RandomGenerator rng) {
        const uint hash = collision_key(gidx, level);
        return get(buffer_index(hash), hash);
    }

    Optional<T>
    get_level(const uint level, const float3 pos, const float3 normal, inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        return get_grid_idx_at_level(gidx, level, normal, rng);
    }

    Optional<T> get_level(const uint level,
                          const float3 pos,
                          const float3 normal,
                          inout RandomGenerator rng,
                          out uint matched_index) {
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        return get_grid_idx_at_level(gidx, level, normal, rng, matched_index);
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

    Optional<T> distributed_get(const float3 pos,
                                const float3 normal,
                                const float3 cam_pos,
                                inout RandomGenerator rng,
                                out uint level,
                                out uint matched_index) {
        level = distributed_level_for_pos(pos, cam_pos, rng);
        return get_level(level, pos, normal, rng, matched_index);
    }

    Optional<T> distributed_get(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        const uint level = distributed_level_for_pos(pos, cam_pos, rng);
        return get_level(level, pos, rng);
    }

    void _update_at_level<let has_normal : bool>(const uint level,
                                                 const float3 pos,
                                                 const float3 normal,
                                                 const float3 prev_cam_pos,
                                                 const uint frame,
                                                 HashGridUpdateFunction<T> update,
                                                 inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);

        uint hash;
        if (has_normal)
            hash = collision_key(gidx, normal, level);
        else
            hash = collision_key(gidx, level);
        const uint base = buffer_index(hash);

        // Evict least-recently-touched slot.
        // nice: an uninitialized slot might have a timestamp in the future, so its unsigned age
        // wraps and is reclaimed first (self-healing).
        uint target = base;
        uint max_age = 0;
        bool matched = false;
        for (uint i = 0; i < PROBE_COUNT; i++) {
            uint idx = base + i;
            if (idx >= BUFFER_SIZE)
                idx -= BUFFER_SIZE;
            let probe = load_key(idx);
            if (keys_match(probe.x, hash)) {
                target = idx;
                matched = true;
                break;
            }
            const uint age = frame - (probe.y & ~MERIAN_HASH_GRID_LOCK_BIT);
            if (age >= max_age) {
                max_age = age;
                target = idx;
            }
        }

        // we use frame so we have it to evict the reast recently once and also to unlock all after
        // the frame implicitly.
        const uint locked = frame | MERIAN_HASH_GRID_LOCK_BIT;
        uint old;
        lock_exchange(target, locked, old);
        if (old == locked) {
            return;
        }

        var t = data[target].t;

        if (!matched) {
            const uint prev_level = target_level_for_pos(pos, prev_cam_pos, rng);
            if (level != prev_level) {
                Optional<T> r;
                if (has_normal)
                    r = get_level(prev_level, pos, normal, rng);
                else
                    r = get_level(prev_level, pos, rng);
                if (let v = r)
                    t = v;
            }
        }

        update.update(t);

        store_slot(target, uint2(hash, frame & ~MERIAN_HASH_GRID_LOCK_BIT), t);
    }

    void update(const float3 pos,
                const float3 normal,
                const float3 cam_pos,
                const float3 prev_cam_pos,
                const uint frame,
                HashGridUpdateFunction<T> update,
                inout RandomGenerator rng) {
        _update_at_level<true>(target_level_for_pos(pos, cam_pos, rng), pos, normal, prev_cam_pos,
                               frame, update, rng);
    }

    void update(const float3 pos,
                const float3 cam_pos,
                const float3 prev_cam_pos,
                const uint frame,
                HashGridUpdateFunction<T> update,
                inout RandomGenerator rng) {
        _update_at_level<false>(target_level_for_pos(pos, cam_pos, rng), pos, float3(0),
                                prev_cam_pos, frame, update, rng);
    }

    void distributed_update(const float3 pos,
                            const float3 normal,
                            const float3 cam_pos,
                            const float3 prev_cam_pos,
                            const uint frame,
                            HashGridUpdateFunction<T> update,
                            inout RandomGenerator rng) {
        _update_at_level<true>(distributed_level_for_pos(pos, cam_pos, rng), pos, normal,
                               prev_cam_pos, frame, update, rng);
    }

    void distributed_update(const float3 pos,
                            const float3 cam_pos,
                            const float3 prev_cam_pos,
                            const uint frame,
                            HashGridUpdateFunction<T> update,
                            inout RandomGenerator rng) {
        _update_at_level<false>(distributed_level_for_pos(pos, cam_pos, rng), pos, float3(0),
                                prev_cam_pos, frame, update, rng);
    }

    // Replaces the slot value instead of accumulating, so no prev-level lookup is needed.
    void _set_at_level<let has_normal : bool>(const uint level,
                                              const float3 pos,
                                              const float3 normal,
                                              const uint frame,
                                              const T t,
                                              inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);

        uint hash;
        if (has_normal)
            hash = collision_key(gidx, normal, level);
        else
            hash = collision_key(gidx, level);
        const uint base = buffer_index(hash);

        // Evict least-recently-touched slot.
        // nice: an uninitialized slot might have a timestamp in the future, so its unsigned age
        // wraps and is reclaimed first (self-healing).
        uint target = base;
        uint max_age = 0;
        for (uint i = 0; i < PROBE_COUNT; i++) {
            uint idx = base + i;
            if (idx >= BUFFER_SIZE)
                idx -= BUFFER_SIZE;
            let probe = load_key(idx);
            if (keys_match(probe.x, hash)) {
                target = idx;
                break;
            }
            const uint age = frame - (probe.y & ~MERIAN_HASH_GRID_LOCK_BIT);
            if (age >= max_age) {
                max_age = age;
                target = idx;
            }
        }

        const uint locked = frame | MERIAN_HASH_GRID_LOCK_BIT;
        uint old;
        lock_exchange(target, locked, old);
        if (old == locked) {
            return;
        }

        store_slot(target, uint2(hash, frame & ~MERIAN_HASH_GRID_LOCK_BIT), t);
    }

    void set_at_level(const uint level,
                      const float3 pos,
                      const float3 normal,
                      const uint frame,
                      const T t,
                      inout RandomGenerator rng) {
        _set_at_level<true>(level, pos, normal, frame, t, rng);
    }

    void set(const float3 pos,
             const float3 normal,
             const float3 cam_pos,
             const uint frame,
             const T t,
             inout RandomGenerator rng) {
        _set_at_level<true>(target_level_for_pos(pos, cam_pos, rng), pos, normal, frame, t, rng);
    }

    void set(const float3 pos,
             const float3 cam_pos,
             const uint frame,
             const T t,
             inout RandomGenerator rng) {
        _set_at_level<false>(target_level_for_pos(pos, cam_pos, rng), pos, float3(0), frame, t,
                             rng);
    }

    void distributed_set(const float3 pos,
                         const float3 normal,
                         const float3 cam_pos,
                         const uint frame,
                         const T t,
                         inout RandomGenerator rng) {
        _set_at_level<true>(distributed_level_for_pos(pos, cam_pos, rng), pos, normal, frame, t,
                            rng);
    }

    void distributed_set(const float3 pos,
                         const float3 cam_pos,
                         const uint frame,
                         const T t,
                         inout RandomGenerator rng) {
        _set_at_level<false>(distributed_level_for_pos(pos, cam_pos, rng), pos, float3(0), frame, t,
                             rng);
    }
}

} // namespace merian

//

//
