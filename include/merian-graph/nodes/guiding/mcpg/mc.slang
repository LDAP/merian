import merian_shaders.utils.grid;
import merian_shaders.utils.random;
import merian_shaders.utils.hash_grid;
import merian_shaders.utils.frame;
__exported import merian_shaders.distributions.distribution;
__exported import merian_shaders.distributions.von_mises_fisher;

using merian;

static const uint16_t ML_MAX_N = 1024;
static const float ML_MIN_ALPHA = 0.01;

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
        const float3 d = get_target() - pos;
        return max(0.0001, dir_guide_prior / dot(d, d));
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

    // Inverse Langevin function, the maximum likelihood concentration of a vMF on S2 whose
    // samples have mean resultant length r. Jedynak 2015.
    float kappa(const float3 pos) {
        const float r = mean_cos(pos);
        return r * (3.0 - 2.6 * r + 0.7 * r * r) / ((1.0 - r) * (1.0 + 0.1 * r));
    }

    // vmf lobe float4(direction, kappa) for a position
    VonMisesFisher3<false> get_vmf(const float3 pos) {
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

struct MCPG<let ADAPTIVE_BUFFER_SIZE : uint,
            let PROBE_COUNT : uint = 2,
            let SPLIT_STORAGE : bool = true,
            let LOCALITY_BITS : uint = 0> {

    HashGrid<MCState, ADAPTIVE_BUFFER_SIZE, PROBE_COUNT, true, true, SPLIT_STORAGE, 2, 1,
             LOCALITY_BITS, true>
        grid;

    MCState load(const float3 pos,
                 const OrthonormalFrame shading_frame,
                 const float3 cam_pos,
                 const float time,
                 inout RandomGenerator rng) {
        // slang defaults initializes none optionals
        var s = grid.distributed_get(pos, shading_frame.get_normal(), cam_pos, rng).value;
        if (s.is_valid()) {
            s.set_w_tgt(s.w_tgt + s.sum_w * (time - s.T) * float3(s.mv));
        } else {
            s.sum_w = 0.0;
        }

        return s;
    }

    // Reports the queried level and where the lobe was read from, invalid if no cell matched.
    MCState load(const float3 pos,
                 const float3 access_normal,
                 const float3 cam_pos,
                 const float time,
                 inout RandomGenerator rng,
                 out HashGridPos grid_pos,
                 out uint level) {
        grid_pos = grid.distributed_find(pos, access_normal, cam_pos, rng, level);
        var s = grid.load(grid_pos).value;
        if (s.is_valid()) {
            s.set_w_tgt(s.w_tgt + s.sum_w * (time - s.T) * float3(s.mv));
        } else {
            s.sum_w = 0.0;
        }

        return s;
    }

    void store(const MCState s,
               const float3 pos,
               const float3 access_normal,
               const float3 cam_pos,
               const uint frame,
               inout RandomGenerator rng) {
        grid.distributed_set(pos, access_normal, cam_pos, frame, s, rng);
    }

    void store_at_level(const MCState s,
                        const float3 pos,
                        const float3 access_normal,
                        const uint level,
                        const uint frame,
                        inout RandomGenerator rng) {
        grid.set_at_level(level, pos, access_normal, frame, s, rng);
    }

    // Missing-light heuristic: invalidate the loaded cell in place.
    void kill(const HashGridPos grid_pos) {
        grid.data[grid_pos.slot].t.sum_w = 0.0;
    }
}
