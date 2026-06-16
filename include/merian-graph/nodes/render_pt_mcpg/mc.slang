__exported import "grid.slang";
import merian_shaders.utils.grid;
import merian_shaders.utils.random;

using merian;

static const uint16_t ML_MAX_N = 1024;
static const float ML_MIN_ALPHA = 0.01;

extern static const uint mc_adaptive_grid_type = GRID_TYPE_EXPONENTIAL;
extern static const float mc_adaptive_grid_tan_alpha_half = 0.003;
extern static const float mc_adaptive_grid_steps_per_unit_size = 6.0;
extern static const float mc_adaptive_grid_min_width = 0.01;
extern static const float mc_adaptive_grid_power = 4.0;
extern static const uint mc_adaptive_buffer_size = 32777259;

extern static const float mc_static_grid_width = 25.3;
extern static const uint mc_static_buffer_size = 800009;

extern static const float dir_guide_prior = 0.2;

// --- GENERAL ---

MCState mc_state_new() {
    MCState s;
    s.w_tgt = float3(0.0);
    s.sum_w = 0.0;
    s.w_cos = 0.0;
    s.mv = half3(0);
    s.T = 0.0;
    s.N = uint16_t(0);
    s.hash = uint16_t(0);
    return s;
}

float3 mc_state_pos(const MCState s) {
    return s.w_tgt / s.sum_w;
}

// returns normalized direction (from pos)
float3 mc_state_dir(const MCState s, const float3 pos) {
    return normalize(mc_state_pos(s) - pos);
}

float mc_state_prior(const MCState s, const float3 pos) {
    const float d = distance(pos, mc_state_pos(s));
    return max(0.0001, dir_guide_prior / (d * d));
}

float mc_state_mean_cos(const MCState s, const float3 pos) {
    const uint n = uint(s.N);
    return clamp((n * n * s.w_cos / s.sum_w) / (n * n + mc_state_prior(s, pos)), 0, 0.9999999);
}

bool mc_light_missing(const MCState s,
                      const float mc_f,
                      const float3 wo,
                      const float3 pos,
                      const float3 next_pos) {
    if (mc_f > 1e-3 * s.sum_w) {
        return false;
    }

    const float cos = dot(wo, mc_state_dir(s, pos));
    if (cos < 0.9 + 0.1 * mc_state_mean_cos(s, pos)) {
        // light might still be there
        return false;
    }

    return true;
}

float mc_state_kappa(const MCState s, const float3 pos) {
    const float r = mc_state_mean_cos(s, pos);
    return (3.0 * r - r * r * r) / (1.0 - r * r);
}

// returns the vmf lobe float4(direction, kappa) for a position
float4 mc_state_get_vmf(const MCState s, const float3 pos) {
    const float4 vmf = float4(mc_state_dir(s, pos), mc_state_kappa(s, pos));

    if (any(isnan(vmf)))
        return float4(0);

    return vmf;
}

void mc_state_reweight(inout MCState s, const float factor) {
    s.sum_w *= factor;
    s.w_tgt *= factor;
    s.w_cos *= factor;
}

// add sample to lobe via maximum likelihood estimator and exponentially weighted average
void mc_state_add_sample(inout MCState s,
                         const float3 pos,         // position where the ray started
                         const float w,            // goodness
                         const float3 target,      // ray hit point
                         const half3 target_mv,
                         const float time) {
    s.N = min(uint16_t(s.N + uint16_t(1)), ML_MAX_N);
    const float alpha = max(1.0 / float(s.N), ML_MIN_ALPHA);

    s.sum_w = lerp(s.sum_w, w, alpha);
    s.w_tgt = lerp(s.w_tgt, w * target, alpha);
    s.w_cos = clamp(lerp(s.w_cos, w * max(0, dot(normalize(target - pos), mc_state_dir(s, pos))),
                         alpha),
                    0, s.sum_w);

    s.mv = target_mv;
    s.T = time;
}

bool mc_state_valid(const MCState s) {
    return s.sum_w > 0.0;
}

// --- ADAPTIVE GRID ---

float mc_target_grid_width(const float3 pos, const float3 cam_pos) {
    return 2 * mc_adaptive_grid_tan_alpha_half * distance(cam_pos, pos);
}

uint mc_adaptive_target_level_for_pos(const float3 pos, const float3 cam_pos) {
    const float w = mc_target_grid_width(pos, cam_pos);
    if (mc_adaptive_grid_type == GRID_TYPE_EXPONENTIAL) {
        return uint(round(mc_adaptive_grid_steps_per_unit_size *
                          log(max(w, mc_adaptive_grid_min_width) / mc_adaptive_grid_min_width) /
                          log(mc_adaptive_grid_power)));
    } else {
        return uint(round(mc_adaptive_grid_steps_per_unit_size *
                          pow(max(w - mc_adaptive_grid_min_width, 0), 1 / mc_adaptive_grid_power)));
    }
}

uint mc_adaptive_level_for_pos(const float3 pos, const float3 cam_pos, const float random) {
    return mc_adaptive_target_level_for_pos(pos, cam_pos) + uint(-log2(1.0 - random));
}

float mc_grid_width_for_level(const uint level) {
    if (mc_adaptive_grid_type == GRID_TYPE_EXPONENTIAL) {
        return mc_adaptive_grid_min_width *
               pow(mc_adaptive_grid_power, level / mc_adaptive_grid_steps_per_unit_size);
    } else {
        return pow(level / mc_adaptive_grid_steps_per_unit_size, mc_adaptive_grid_power) +
               mc_adaptive_grid_min_width;
    }
}

void mc_adaptive_buffer_index(const float3 pos,
                              const float3 normal,
                              const float3 cam_pos,
                              inout RandomGenerator rng,
                              out uint buffer_index,
                              out uint16_t hash) {
    const uint level = mc_adaptive_level_for_pos(pos, cam_pos, rng.next_float());
    const int3 grid_idx = grid_idx_interpolate(pos, mc_grid_width_for_level(level), rng.next_float());
    buffer_index = hash_grid_normal_level(uint3(grid_idx), normal, level, mc_adaptive_buffer_size);
    hash = uint16_t(hash2_grid_level(grid_idx, level));
}

void mc_adaptive_finalize_load(inout MCState s, const uint16_t hash, const float time) {
    if (s.sum_w < 0 || hash != s.hash) {
        s.sum_w = 0.0;
    }
    s.w_tgt += s.sum_w * (time - s.T) * float3(s.mv);
}

void mc_adaptive_load(RWStructuredBuffer<MCState> mc_states,
                      out MCState s,
                      const float3 pos,
                      const float3 normal,
                      const float3 cam_pos,
                      const float time,
                      inout RandomGenerator rng) {
    uint buffer_index;
    uint16_t hash;
    mc_adaptive_buffer_index(pos, normal, cam_pos, rng, buffer_index, hash);
    s = mc_states[buffer_index];
    mc_adaptive_finalize_load(s, hash, time);
}

void mc_adaptive_save(RWStructuredBuffer<MCState> mc_states,
                      MCState s,
                      const float3 pos,
                      const float3 normal,
                      const float3 cam_pos,
                      inout RandomGenerator rng) {
    uint buffer_index;
    uint16_t hash;
    mc_adaptive_buffer_index(pos, normal, cam_pos, rng, buffer_index, hash);

    s.hash = hash;
    mc_states[buffer_index] = s;
}

// --- STATIC GRID ---

void mc_static_buffer_index(const float3 pos,
                            inout RandomGenerator rng,
                            out uint buffer_index,
                            out uint16_t hash) {
    const int3 grid_idx = grid_idx_interpolate(pos, mc_static_grid_width, rng.next_float());
    buffer_index = hash_grid(uint3(grid_idx), mc_static_buffer_size) + mc_adaptive_buffer_size;
    hash = uint16_t(hash2_grid(grid_idx));
}

void mc_static_finalize_load(inout MCState s, const uint16_t hash, const float time) {
    if (s.sum_w < 0 || hash != s.hash) {
        s.sum_w = 0.0;
    }
    s.w_tgt += s.sum_w * (time - s.T) * float3(s.mv);
}

void mc_static_finalize_load(inout MCState s,
                             const uint16_t hash,
                             const float3 pos,
                             const float3 normal,
                             const float time) {
    if (s.sum_w <= 0 || hash != s.hash || dot(normal, mc_state_dir(s, pos)) <= 0.0) {
        s.sum_w = 0.0;
    }
    s.w_tgt += s.sum_w * (time - s.T) * float3(s.mv);
}

void mc_static_load(RWStructuredBuffer<MCState> mc_states,
                    out MCState s,
                    const float3 pos,
                    const float time,
                    inout RandomGenerator rng) {
    uint buffer_index;
    uint16_t hash;
    mc_static_buffer_index(pos, rng, buffer_index, hash);
    s = mc_states[buffer_index];
    mc_static_finalize_load(s, hash, time);
}

void mc_static_load(RWStructuredBuffer<MCState> mc_states,
                    out MCState s,
                    const float3 pos,
                    const float3 normal,
                    const float time,
                    inout RandomGenerator rng) {
    uint buffer_index;
    uint16_t hash;
    mc_static_buffer_index(pos, rng, buffer_index, hash);
    s = mc_states[buffer_index];
    mc_static_finalize_load(s, hash, pos, normal, time);
}

void mc_static_save(RWStructuredBuffer<MCState> mc_states,
                    MCState s,
                    const float3 pos,
                    inout RandomGenerator rng) {
    uint buffer_index;
    uint16_t hash;
    mc_static_buffer_index(pos, rng, buffer_index, hash);

    s.hash = hash;
    mc_states[buffer_index] = s;
}
