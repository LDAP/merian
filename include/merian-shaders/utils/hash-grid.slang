import merian_shaders.utils.encoding;
import merian_shaders.utils.envmap;
import merian_shaders.utils.grid;
import merian_shaders.utils.hash;
import merian_shaders.utils.morton;
import merian_shaders.utils.random;

namespace merian {

static const uint MERIAN_HASH_GRID_LOCK_BIT = 0x80000000u;
// Shifts the base level for a unit, allows resuolution up to ~2^-31 units
static const float MERIAN_HASH_GRID_LEVEL_OFFSET = 32.0;
static const uint MERIAN_HASH_GRID_INVALID_SLOT = 0xFFFFFFFFu;

interface HashGridUpdateFunction<typename T> {
    void update(inout T t);
}

// Where a cell lives: the slot it occupies and the key stored there.
struct HashGridPos {
    uint slot;
    uint key;

    __init() {
        slot = MERIAN_HASH_GRID_INVALID_SLOT;
        key = 0;
    }

    __init(const uint slot, const uint key) {
        this.slot = slot;
        this.key = key;
    }

    bool is_valid() {
        return slot != MERIAN_HASH_GRID_INVALID_SLOT;
    }
}

struct HashGridSlot<typename T, let SPLIT_STORAGE : bool> {
    uint2 key[SPLIT_STORAGE ? 0 : 1];
    T t;
}

// LOCK_FREE: writers skip the slot lock
struct HashGrid<typename T,
                let BUFFER_SIZE : uint,
                let PROBE_COUNT : uint = 4,
                let STOCHASTIC_INTERPOLATION : bool = false,
                let LEVEL_STOCHASTIC_INTERPOLATION : bool = false,
                let SPLIT_STORAGE : bool = true,
                let GRID_POWER : uint = 2,
                let GRID_STEPS_PER_UNIT_SIZE : uint = 1,
                let LOCALITY_BITS : uint = 0,
                let LOCK_FREE : bool = false> {

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

    // The lattice math only needs pos in cell widths, so no access has to divide.
    float grid_inv_width_for_level(const uint level) {
        return (0.5 / grid_tan_alpha_half) *
               pow(GRID_POWER, MERIAN_HASH_GRID_LEVEL_OFFSET -
                                   (float(level) - grid_level_bias) /
                                       float(GRID_STEPS_PER_UNIT_SIZE));
    }

    int3 grid_idx(const float3 pos, const float inv_width, inout RandomGenerator rng) {
        // Offset the lattice by half a cell so that levels of different width don't share vertices
        // (Müller et al., "Instant Neural Graphics Primitives", Appendix A.)
        const float3 staggered = pos * inv_width + 0.5;
        if (STOCHASTIC_INTERPOLATION)
            return grid_idx_interpolate_scaled(staggered, rng.next_float());
        return grid_idx_lower_scaled(staggered);
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

    // The slot a cell probes from and the key stored there to detect collisions.
    HashGridPos get_hash_grid_pos(const int3 gidx, const uint level_normal) {
        const uint tile_count = BUFFER_SIZE >> (3u * LOCALITY_BITS);
        uint slot;
        uint key;
        if (LOCALITY_BITS == 0 || tile_count == 0) {
            key = pcg4d16(uint4(gidx, level_normal));
            slot = key % BUFFER_SIZE;
        } else {
            // LOCALITY_BITS = L gives each 2^L-wide lattice tile (per level and normal bin)
            // 8^L contiguous slots in Morton order, so neighbouring cells share
            // cache lines instead of scattering onto one line each.
            const uint cell_mask = (1u << LOCALITY_BITS) - 1u;
            const uint tile_hash = pcg4d16(uint4(gidx >> LOCALITY_BITS, level_normal));
            const uint cell = morton_encode3d(uint3(gidx) & cell_mask);
            // The odd multiplier is injective, so cells within a tile never share a key.
            key = tile_hash ^ (cell * 0x9E3779B9u);
            slot = ((tile_hash % tile_count) << (3u * LOCALITY_BITS)) + cell;
        }
        // Make blocks of PROBE_COUNT slots compute the same starting index into the buffer
        // for coalescing memory accesses, i.e. coherent probing accross the warp.
        if ((PROBE_COUNT & (PROBE_COUNT - 1u)) == 0)
            slot &= ~(PROBE_COUNT - 1u);
        return HashGridPos(slot, key);
    }

    HashGridPos get_hash_grid_pos(const int3 gidx, const uint level, const float3 normal) {
        return get_hash_grid_pos(gidx, level_normal_word(level, normal));
    }

    // Spatial (normal-less) slot, for debug coloring by cell.
    uint buffer_index(const int3 gidx, const uint level) {
        return get_hash_grid_pos(gidx, level_normal_word(level)).slot;
    }

    bool keys_match(const uint stored, const uint query) {
        return stored == query;
    }

    // Linear-probes PROBE_COUNT slots from pos for its key. The result names the slot the cell
    // occupies, or is invalid if the grid does not hold it.
    HashGridPos find(const HashGridPos pos) {
        uint slot = pos.slot;
        for (uint i = 0; i < PROBE_COUNT; i++) {
            if (keys_match(load_hash(slot), pos.key)) {
                return HashGridPos(slot, pos.key);
            }

            slot++;
            if (slot >= BUFFER_SIZE)
                slot -= BUFFER_SIZE;
        }

        return HashGridPos();
    }

    Optional<T> load(const HashGridPos pos) {
        if (pos.is_valid())
            return data[pos.slot].t;
        return none;
    }

    // Debug only: octahedral bin of the normal if an exact slot matches at the target level.
    uint debug_selected_bin(const float3 pos,
                            const float3 normal,
                            const float3 cam_pos,
                            inout RandomGenerator rng) {
        const uint level = target_level_for_pos(pos, cam_pos, rng);
        const int3 gidx = grid_idx(pos, grid_inv_width_for_level(level), rng);
        if (find(get_hash_grid_pos(gidx, level, normal)).is_valid())
            return normal_bin(normal);
        return MERIAN_HASH_GRID_INVALID_SLOT;
    }

    HashGridPos
    find_level(const uint level, const float3 pos, const float3 normal, inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_inv_width_for_level(level), rng);
        return find(get_hash_grid_pos(gidx, level, normal));
    }

    HashGridPos find_level(const uint level, const float3 pos, inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_inv_width_for_level(level), rng);
        return find(get_hash_grid_pos(gidx, level_normal_word(level)));
    }

    Optional<T>
    get_level(const uint level, const float3 pos, const float3 normal, inout RandomGenerator rng) {
        return load(find_level(level, pos, normal, rng));
    }

    Optional<T> get_level(const uint level, const float3 pos, inout RandomGenerator rng) {
        return load(find_level(level, pos, rng));
    }

    Optional<T>
    get(const float3 pos, const float3 normal, const float3 cam_pos, inout RandomGenerator rng) {
        return get_level(target_level_for_pos(pos, cam_pos, rng), pos, normal, rng);
    }

    Optional<T> get(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        return get_level(target_level_for_pos(pos, cam_pos, rng), pos, rng);
    }

    // found is the cell at the level the current camera implies, so an update may reuse it. It is
    // invalid when the value came from the previous camera's level instead, which an update at
    // this camera must not write to.
    Optional<T> get(const float3 pos,
                    const float3 normal,
                    const float3 cam_pos,
                    const float3 prev_cam_pos,
                    inout RandomGenerator rng,
                    out HashGridPos found) {
        const uint level = target_level_for_pos(pos, cam_pos, rng);
        found = find_level(level, pos, normal, rng);
        if (found.is_valid()) {
            return load(found);
        }

        if (any(cam_pos != prev_cam_pos)) {
            const uint prev_level = target_level_for_pos(pos, prev_cam_pos, rng);
            if (level != prev_level) {
                return get_level(prev_level, pos, normal, rng);
            }
        }

        return none;
    }

    Optional<T> get(const float3 pos,
                    const float3 normal,
                    const float3 cam_pos,
                    const float3 prev_cam_pos,
                    inout RandomGenerator rng) {
        HashGridPos found;
        return get(pos, normal, cam_pos, prev_cam_pos, rng, found);
    }

    HashGridPos distributed_find(const float3 pos,
                                 const float3 normal,
                                 const float3 cam_pos,
                                 inout RandomGenerator rng,
                                 out uint level) {
        level = distributed_level_for_pos(pos, cam_pos, rng);
        return find_level(level, pos, normal, rng);
    }

    Optional<T> distributed_get(const float3 pos,
                                const float3 normal,
                                const float3 cam_pos,
                                inout RandomGenerator rng) {
        return get_level(distributed_level_for_pos(pos, cam_pos, rng), pos, normal, rng);
    }

    Optional<T> distributed_get(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        return get_level(distributed_level_for_pos(pos, cam_pos, rng), pos, rng);
    }

    void _update_at_level<let has_normal : bool>(const uint level,
                                                 const float3 pos,
                                                 const float3 normal,
                                                 const float3 prev_cam_pos,
                                                 const uint frame,
                                                 HashGridUpdateFunction<T> update,
                                                 inout RandomGenerator rng) {
        const int3 gidx = grid_idx(pos, grid_inv_width_for_level(level), rng);

        const uint level_normal =
            has_normal ? level_normal_word(level, normal) : level_normal_word(level);
        const HashGridPos cell = get_hash_grid_pos(gidx, level_normal);
        const uint base = cell.slot;
        const uint hash = cell.key;

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
        if (!LOCK_FREE) {
            const uint locked = frame | MERIAN_HASH_GRID_LOCK_BIT;
            uint old;
            lock_exchange(target, locked, old);
            if (old == locked) {
                return;
            }
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

    // Does not probe, so grid_pos must still locate the cell.
    void update_at(const HashGridPos grid_pos,
                      const uint frame,
                      HashGridUpdateFunction<T> update) {
        if (!LOCK_FREE) {
            const uint locked = frame | MERIAN_HASH_GRID_LOCK_BIT;
            uint old;
            lock_exchange(grid_pos.slot, locked, old);
            if (old == locked) {
                return;
            }
        }

        var t = data[grid_pos.slot].t;
        update.update(t);
        store_slot(grid_pos.slot, uint2(grid_pos.key, frame & ~MERIAN_HASH_GRID_LOCK_BIT), t);
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
        const int3 gidx = grid_idx(pos, grid_inv_width_for_level(level), rng);

        const uint level_normal =
            has_normal ? level_normal_word(level, normal) : level_normal_word(level);
        const HashGridPos cell = get_hash_grid_pos(gidx, level_normal);
        const uint base = cell.slot;
        const uint hash = cell.key;

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

        if (!LOCK_FREE) {
            const uint locked = frame | MERIAN_HASH_GRID_LOCK_BIT;
            uint old;
            lock_exchange(target, locked, old);
            if (old == locked) {
                return;
            }
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
