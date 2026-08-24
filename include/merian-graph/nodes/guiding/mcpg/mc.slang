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
// Standard normal quantile of the lobe width's upper confidence limit, see width_upper_limit.
// 0 samples the maximum likelihood width instead.
extern static const float mc_conf_z = 1.6449;
// Width an undetermined chain falls back to. 1 - 2/3 is the mean resultant length deficit of a
// cosine lobe, so such a chain proposes what the BSDF would and costs nothing.
static const float ML_MAX_ONE_MINUS_COS = 1.0 / 3.0;

struct MCState {
    float3 w_tgt;
    float sum_w;
    // weighted mean squared chord to the mean direction, which is exactly 2 (1 - cos)
    float w_chord2;

    half3 mv;
    uint16_t N;

    float T;

    __init() {
        w_tgt = float3(0.0);
        sum_w = 0.0;
        w_chord2 = 0.0;
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

    // Upper confidence limit of the width, over the width itself. 1 - cos is exponentially
    // distributed in the concentration, so n samples give a Gamma estimate and its limit is a
    // chi-square quantile away; one degree of freedom goes to the axis the spread is measured
    // against, so a single sample leaves the width undetermined. Wilson-Hilferty for the quantile,
    // which is within 2.4 % for n >= 9 and diverges below n ~ 3, where the clamp decides instead.
    static float width_upper_limit(const float u, const float n) {
        const float k = 2.0 * (n - 1.0);
        if (k <= 0.0)
            return ML_MAX_ONE_MINUS_COS;
        const float t = 2.0 / (9.0 * k);
        const float g = 1.0 - t - mc_conf_z * sqrt(t);
        if (g <= 0.0)
            return ML_MAX_ONE_MINUS_COS;
        return min(u * (2.0 * n / k) / (g * g * g), ML_MAX_ONE_MINUS_COS);
    }

    // 1 - mean resultant length. Kept as the primary quantity: it is what the concentration
    // depends on, and the half that float resolves once the lobe is tight.
    float one_minus_mean_cos(const float3 pos) {
        const float n = float(N);
        const float p = prior(pos);
        float u = 0.5 * w_chord2 / sum_w;
        if (mc_conf_z > 0.0) {
            // samples behind the fit; the exponentially weighted average saturates at 2 / alpha - 1
            const float n_eff = min(2.0 * n - 1.0, 2.0 / ML_MIN_ALPHA - 1.0);
            u = width_upper_limit(u, n_eff);
        }
        // shrink r towards 0 by n^2 / (n^2 + p), i.e. grow u by the complement
        return clamp(u + (1.0 - u) * p / (n * n + p), 1e-7, 1.0);
    }

    float mean_cos(const float3 pos) {
        return 1.0 - one_minus_mean_cos(pos);
    }

    bool light_missing_heuristic(const float mc_f,
                                 const float3 wo,
                                 const float3 pos,
                                 const float3 next_pos) {
        if (mc_f > 1e-3 * sum_w) {
            return false;
        }

        // the sample has to have probed the lobe's core to say anything about the target
        return 1.0 - dot(wo, get_dir(pos)) <= 0.1 * one_minus_mean_cos(pos);
    }

    // Inverse Langevin function, the maximum likelihood concentration of a vMF on S2 whose
    // samples have mean resultant length r = 1 - u. Jedynak 2015. Carried in u because
    // kappa ~ 1/u, so u holds the concentration to full relative precision where r saturates.
    float kappa(const float3 pos) {
        const float u = one_minus_mean_cos(pos);
        const float r = 1.0 - u;
        return r * (3.0 - 2.6 * r + 0.7 * r * r) / (u * (1.1 - 0.1 * u));
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
        w_chord2 *= factor;
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
        const float3 chord = normalize(target - pos) - get_dir(pos);
        w_chord2 = lerp(w_chord2, w * dot(chord, chord), alpha);

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
