import merian_shaders.utils.encoding;
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

// NORMAL_BITS: the bits for normal bin count, neighboring bins share their entries. Set to 0 to
// ignore the normal. 6 => ~22° cells, 8 => ~11° cells (more hurts the hash quality)
struct HashGrid<typename T,
                let BUFFER_SIZE : uint,
                let PROBE_COUNT : uint = 4,
                let STOCHASTIC_INTERPOLATION : bool = false,
                let LEVEL_STOCHASTIC_INTERPOLATION : bool = false,
                let NORMAL_BITS : uint = 0,
                let GRID_POWER : uint = 2,
                let GRID_STEPS_PER_UNIT_SIZE : uint = 1> {
    RWStructuredBuffer<HashGridVertex<T>, ScalarDataLayout> buffer;

    float grid_tan_alpha_half;
    // SHARC-style LOD bias: shifts the level quantization / near-camera detail (0 = neutral).
    float grid_level_bias;
    // Effective spatial dimensionality the distributed levels are spread over (2 = surface,
    // 3 = volume). Keeps the expected threads per cell roughly constant; smaller widens the spread.
    float grid_distribution_dimension;

    uint target_level_for_pos(const float3 pos, const float3 cam_pos, inout RandomGenerator rng) {
        const float3 d = cam_pos - pos;
        const float dist2 = max(dot(d, d), 1e-12);
        const float level =
            max(GRID_STEPS_PER_UNIT_SIZE * 0.5 * log(dist2) / log(GRID_POWER) + grid_level_bias, 1.0);
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
               pow(GRID_POWER, (float(level) - grid_level_bias) / float(GRID_STEPS_PER_UNIT_SIZE));
    }

    int3 grid_idx(const float3 pos, const float width, inout RandomGenerator rng) {
        // Offset the lattice by half a cell so that levels of different width don't share vertices
        // (Müller et al., "Instant Neural Graphics Primitives", Appendix A.)
        const float3 staggered = pos + 0.5 * width;
        if (STOCHASTIC_INTERPOLATION)
            return grid_idx_interpolate(staggered, width, rng.next_float());
        return grid_idx_lower(staggered, width);
    }

    uint hash_key(const int3 gidx, const uint level) {
        return xxhash32(uint4(gidx, level));
    }

    uint buffer_index(const int3 gidx, const uint level) {
        return hash_key(gidx, level) % BUFFER_SIZE;
    }

    uint normal_bin_mask() {
        return NORMAL_BITS == 0 ? 0u : ((1u << NORMAL_BITS) - 1u);
    }

    // Octahedral cell of the normal, packed into NORMAL_BITS (half the bits per oct axis).
    // Bias off the axes so an axis-aligned normal doesn't sit exactly on a cell edge and flicker.
    uint normal_bin(const float3 normal) {
        if (NORMAL_BITS == 0)
            return 0u;
        const uint per_axis = NORMAL_BITS / 2;
        const uint res = 1u << per_axis;
        const float2 oct = octahedral_project(normal + 1e-3) * 0.5 + 0.5; // [0, 1]^2
        const int2 cell = clamp(int2(floor(oct * res)), int2(0), int2(res - 1));
        return (uint(cell.x) << per_axis) | uint(cell.y);
    }

    // Spatial hash in the upper bits, octahedral normal cell in the low NORMAL_BITS.
    uint collision_key(const int3 gidx, const float3 normal, const uint level) {
        const uint spatial = pcg4d16(uint4(gidx, level));
        return (spatial & ~normal_bin_mask()) | normal_bin(normal);
    }

    uint collision_key(const int3 gidx, const uint level) {
        return pcg4d16(uint4(gidx, level)) & ~normal_bin_mask();
    }

    // Ignores the normal bins, so a normal-less query matches an entry stored with any normal.
    bool keys_match_spatial(const uint stored, const uint query) {
        return (stored & ~normal_bin_mask()) == (query & ~normal_bin_mask());
    }

    // Match the spatial hash exactly and the octahedral cell within a 1-cell (Chebyshev) tolerance,
    // so normals up to ~one bin apart share a slot without hard binning seams.
    bool keys_match(const uint stored, const uint query) {
        if (NORMAL_BITS == 0)
            return stored == query;
        if ((stored & ~normal_bin_mask()) != (query & ~normal_bin_mask()))
            return false;
        const uint per_axis = NORMAL_BITS / 2;
        const uint axis_mask = (1u << per_axis) - 1u;
        const int2 s = int2((stored >> per_axis) & axis_mask, stored & axis_mask);
        const int2 q = int2((query >> per_axis) & axis_mask, query & axis_mask);
        const int2 d = abs(s - q);
        return max(d.x, d.y) <= 1;
    }

    // Linear-probe PROBE_COUNT slots starting at buffer_index for a matching hash.
    Optional<T> get(uint buffer_index, const uint hash) {
        for (uint i = 0; i < PROBE_COUNT; i++) {
            let vtx = buffer[buffer_index];

            if (keys_match(vtx.hash, hash)) {
                return vtx.data;
            }

            buffer_index++;
            if (buffer_index >= BUFFER_SIZE)
                buffer_index -= BUFFER_SIZE;
        }

        return none;
    }

    // Linear-probe ignoring the normal bins, matching an entry stored with any normal.
    Optional<T> get_any_normal(uint buffer_index, const uint hash) {
        for (uint i = 0; i < PROBE_COUNT; i++) {
            let vtx = buffer[buffer_index];

            if (keys_match_spatial(vtx.hash, hash)) {
                return vtx.data;
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
            let vtx = buffer[buffer_index];

            if (keys_match(vtx.hash, hash)) {
                matched_index = buffer_index;
                return vtx.data;
            }

            buffer_index++;
            if (buffer_index >= BUFFER_SIZE)
                buffer_index -= BUFFER_SIZE;
        }

        matched_index = 0xFFFFFFFFu;
        return none;
    }

    // Debug only: octahedral bin of the slot a normal query actually reads at its target level,
    // or 0xFFFFFFFF on miss. Compare against normal_bin(normal) to see tolerance merges.
    uint debug_selected_bin(const float3 pos,
                            const float3 normal,
                            const float3 cam_pos,
                            inout RandomGenerator rng) {
        const uint level = target_level_for_pos(pos, cam_pos, rng);
        const int3 gidx = grid_idx(pos, grid_width_for_level(level), rng);
        const uint hash = collision_key(gidx, normal, level);
        uint bi = buffer_index(gidx, level);
        for (uint i = 0; i < PROBE_COUNT; i++) {
            let vtx = buffer[bi];
            if (keys_match(vtx.hash, hash))
                return vtx.hash & normal_bin_mask();
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
        return get(buffer_index(gidx, level), hash);
    }

    Optional<T> get_grid_idx_at_level(const int3 gidx,
                                      const uint level,
                                      const uint hash,
                                      inout RandomGenerator rng,
                                      out uint matched_index) {
        return get(buffer_index(gidx, level), hash, matched_index);
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
        return get_any_normal(buffer_index(gidx, level), collision_key(gidx, level));
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
            if (keys_match(probe.hash, hash)) {
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

        if (!keys_match(vtx.hash, hash)) {
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
        bool matched = false;
        for (uint i = 0; i < PROBE_COUNT; i++) {
            uint idx = base + i;
            if (idx >= BUFFER_SIZE)
                idx -= BUFFER_SIZE;
            let probe = buffer[idx];
            if (keys_match(probe.hash, hash)) {
                target = idx;
                matched = true;
                break;
            }
            const uint age = frame - (probe.lock & ~MERIAN_HASH_GRID_LOCK_BIT);
            if (age >= max_age) {
                max_age = age;
                target = idx;
            }
        }

        const uint locked = frame | MERIAN_HASH_GRID_LOCK_BIT;
        uint old;
        InterlockedExchange(buffer[target].lock, locked, old);
        if (old == locked) {
            return;
        }

        var vtx = buffer[target];
        // keep the anchoring bin on a tolerance match so the slot's oct cell doesn't drift
        if (!matched)
            vtx.hash = hash;
        vtx.data = t;
        vtx.lock = frame & ~MERIAN_HASH_GRID_LOCK_BIT;
        buffer[target] = vtx;
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
