import merian_shaders.utils.grid;
import merian_shaders.utils.random;
import merian_shaders.utils.hash_grid;
__exported import merian_shaders.distributions.distribution;
import merian_shaders.distributions.von_mises_fisher;

using merian;

static const uint16_t ML_MAX_N = 1024;
static const float ML_MIN_ALPHA = 0.01;

static const uint MC_INVALID_INDEX = 0xFFFFFFFFu;

extern static const float dir_guide_prior = 0.2;

struct MCState {
    float3 w_tgt;
    float sum_w;
    float w_cos;

    half3 mv;
    uint16_t N;

    float T;

    __init() {
        w_tgt = float3(0.0);
        sum_w = 0.0;
        w_cos = 0.0;
        mv = half3(0);
        T = 0.0;
        N = uint16_t(0);
    }

    float3 get_target() {
        return w_tgt / sum_w;
    }

    // normalized direction towards the target
    float3 get_dir(const float3 from) {
        return normalize(get_target() - from);
    }

    float prior(const float3 pos) {
        const float d = distance(pos, get_target());
        return max(0.0001, dir_guide_prior / (d * d));
    }

    float mean_cos(const float3 pos) {
        const uint n = uint(N);
        return clamp((n * n * w_cos / sum_w) / (n * n + prior(pos)), 0, 0.9999999);
    }

    bool light_missing_heuristic(const float mc_f,
                                 const float3 wo,
                                 const float3 pos,
                                 const float3 next_pos) {
        if (mc_f > 1e-3 * sum_w) {
            return false;
        }

        const float cos = dot(wo, get_dir(pos));
        if (cos < 0.9 + 0.1 * mean_cos(pos)) {
            return false;
        }

        return true;
    }

    float kappa(const float3 pos) {
        const float r = mean_cos(pos);
        return (3.0 * r - r * r * r) / (1.0 - r * r);
    }

    // vmf lobe float4(direction, kappa) for a position
    Distribution<float3> get_vmf(const float3 pos) {
        const float3 dir = get_dir(pos);
        const float k = kappa(pos);

        if (isnan(k) || any(isnan(dir)))
            return VonMisesFisher3<false>(0, float3(0));

        return VonMisesFisher3<false>(k, dir);
    }

    [mutating]
    void reweight(const float factor) {
        sum_w *= factor;
        w_tgt *= factor;
        w_cos *= factor;
    }

    // add sample to lobe via maximum likelihood estimator and exponentially weighted average
    [mutating]
    void add_sample(const float3 pos,    // position where the ray started
                    const float w,       // goodness
                    const float3 target, // ray hit point
                    const half3 target_mv,
                    const float time) {
        N = min(uint16_t(N + uint16_t(1)), ML_MAX_N);
        const float alpha = max(1.0 / float(N), ML_MIN_ALPHA);

        sum_w = lerp(sum_w, w, alpha);
        w_tgt = lerp(w_tgt, w * target, alpha);
        w_cos = clamp(lerp(w_cos, w * max(0, dot(normalize(target - pos), get_dir(pos))), alpha), 0,
                      sum_w);
        // mc_state.w_cos = min(length(mix(mc_state.w_cos * mc_state_dir(mc_state, pos), w *
        // normalize(target - pos), alpha)), mc_state.sum_w);

        mv = target_mv;
        T = time;
    }

    bool is_valid() {
        return sum_w > 0.0;
    }

    [mutating]
    void set_w_tgt(const float3 w_tgt) {
        this.w_tgt = w_tgt;
    }
};

struct MCPG<let ADAPTIVE_BUFFER_SIZE : uint, let NORMAL_BITS : uint = 8> {
    HashGrid<MCState, ADAPTIVE_BUFFER_SIZE, 2, true, true, NORMAL_BITS> grid;

    MCState load(const float3 pos,
                 const float3 normal,
                 const float3 cam_pos,
                 const float time,
                 inout RandomGenerator rng) {
        // slang defaults initializes none optionals
        var s = grid.distributed_get(pos, normal, cam_pos, rng).value;
        if (s.is_valid()) {
            s.set_w_tgt(s.w_tgt + s.sum_w * (time - s.T) * float3(s.mv));
        } else {
            s.sum_w = 0.0;
        }

        return s;
    }

    // Reports the queried level and the exact buffer slot the lobe was read from
    // (MC_INVALID_INDEX if no slot matched).
    MCState load(const float3 pos,
                 const float3 normal,
                 const float3 cam_pos,
                 const float time,
                 inout RandomGenerator rng,
                 out uint buffer_index,
                 out uint level) {
        var s = grid.distributed_get(pos, normal, cam_pos, rng, level, buffer_index).value;
        if (s.is_valid()) {
            s.set_w_tgt(s.w_tgt + s.sum_w * (time - s.T) * float3(s.mv));
        } else {
            s.sum_w = 0.0;
        }

        return s;
    }

    void store(const MCState s,
               const float3 pos,
               const float3 normal,
               const float3 cam_pos,
               const uint frame,
               inout RandomGenerator rng) {
        grid.distributed_set(pos, normal, cam_pos, frame, s, rng);
    }

    void store_at_level(const MCState s,
                        const float3 pos,
                        const float3 normal,
                        const uint level,
                        const uint frame,
                        inout RandomGenerator rng) {
        grid.set_at_level(level, pos, normal, frame, s, rng);
    }

    // Missing-light heuristic: invalidate the exact loaded slot in place.
    void kill(const uint buffer_index) {
        grid.buffer[buffer_index].data.sum_w = 0.0;
    }
}

// --- ADAPTIVE GRID ---

// void mc_adaptive_buffer_index(const float3 pos,
//                               const float3 normal,
//                               const float3 cam_pos,
//                               inout RandomGenerator rng,
//                               out uint buffer_index,
//                               out uint16_t hash) {
//     const uint level = mc_adaptive_level_for_pos(pos, cam_pos, rng.next_float());
//     const int3 grid_idx =
//         grid_idx_interpolate(pos, mc_grid_width_for_level(level), rng.next_float());
//     buffer_index = hash_grid_normal_level(uint3(grid_idx), normal, level,
//     mc_adaptive_buffer_size); hash = uint16_t(hash2_grid_level(grid_idx, level));
// }

// void mc_adaptive_finalize_load(inout MCState s, const uint16_t hash, const float time) {
//     if (s.sum_w < 0 || hash != s.hash) {
//         s.sum_w = 0.0;
//     }
//
// }

// void mc_adaptive_load(RWStructuredBuffer<MCState> mc_states,
//                       out MCState s,
//                       const float3 pos,
//                       const float3 normal,
//                       const float3 cam_pos,
//                       const float time,
//                       inout RandomGenerator rng) {
//     uint buffer_index;
//     uint16_t hash;
//     mc_adaptive_buffer_index(pos, normal, cam_pos, rng, buffer_index, hash);
//     s = mc_states[buffer_index];
//     mc_adaptive_finalize_load(s, hash, time);
// }

// void mc_adaptive_save(RWStructuredBuffer<MCState> mc_states,
//                       MCState s,
//                       const float3 pos,
//                       const float3 normal,
//                       const float3 cam_pos,
//                       inout RandomGenerator rng) {
//     uint buffer_index;
//     uint16_t hash;
//     mc_adaptive_buffer_index(pos, normal, cam_pos, rng, buffer_index, hash);

//     s.hash = hash;
//     mc_states[buffer_index] = s;
// }

// // --- STATIC GRID ---

// void mc_static_buffer_index(const float3 pos,
//                             inout RandomGenerator rng,
//                             out uint buffer_index,
//                             out uint16_t hash) {
//     const int3 grid_idx = grid_idx_interpolate(pos, mc_static_grid_width, rng.next_float());
//     buffer_index = hash_grid(uint3(grid_idx), mc_static_buffer_size) + mc_adaptive_buffer_size;
//     hash = uint16_t(hash2_grid(grid_idx));
// }

// void mc_static_finalize_load(inout MCState s, const uint16_t hash, const float time) {
//     if (s.sum_w < 0 || hash != s.hash) {
//         s.sum_w = 0.0;
//     }
//     s.w_tgt += s.sum_w * (time - s.T) * float3(s.mv);
// }

// void mc_static_finalize_load(
//     inout MCState s, const uint16_t hash, const float3 pos, const float3 normal, const float
//     time) { if (s.sum_w <= 0 || hash != s.hash || dot(normal, s.get_dir(pos)) <= 0.0) {
//         s.sum_w = 0.0;
//     }
//     s.w_tgt += s.sum_w * (time - s.T) * float3(s.mv);
// }

// void mc_static_load(RWStructuredBuffer<MCState> mc_states,
//                     out MCState s,
//                     const float3 pos,
//                     const float time,
//                     inout RandomGenerator rng) {
//     uint buffer_index;
//     uint16_t hash;
//     mc_static_buffer_index(pos, rng, buffer_index, hash);
//     s = mc_states[buffer_index];
//     mc_static_finalize_load(s, hash, time);
// }

// void mc_static_load(RWStructuredBuffer<MCState> mc_states,
//                     out MCState s,
//                     const float3 pos,
//                     const float3 normal,
//                     const float time,
//                     inout RandomGenerator rng) {
//     uint buffer_index;
//     uint16_t hash;
//     mc_static_buffer_index(pos, rng, buffer_index, hash);
//     s = mc_states[buffer_index];
//     mc_static_finalize_load(s, hash, pos, normal, time);
// }

// void mc_static_save(RWStructuredBuffer<MCState> mc_states,
//                     MCState s,
//                     const float3 pos,
//                     inout RandomGenerator rng) {
//     uint buffer_index;
//     uint16_t hash;
//     mc_static_buffer_index(pos, rng, buffer_index, hash);

//     s.hash = hash;
//     mc_states[buffer_index] = s;
// }
