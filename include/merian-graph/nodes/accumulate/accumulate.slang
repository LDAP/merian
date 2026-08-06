#define FILTER_MODE_NEAREST 0
#define FILTER_MODE_STOCHASTIC_BILINEAR 1

// When the reprojected GBuffer (normal/depth) is consulted to validate reuse.
#define GBUFFER_CHECK_ALWAYS 0    // always validate (default)
#define GBUFFER_CHECK_IF_MOVING 1 // skip the check (accept) where the motion vector is zero
#define GBUFFER_CHECK_NEVER 2     // never validate, trust the motion vectors

[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;

[vk::constant_id(2)]
const uint WG_ROUNDED_IRR_SIZE_X = 1;
[vk::constant_id(3)]
const uint WG_ROUNDED_IRR_SIZE_Y = 1;
[vk::constant_id(4)]
const int FILTER_MODE = FILTER_MODE_NEAREST;
[vk::constant_id(5)]
const bool EXTENDED_SEARCH = true;
[vk::constant_id(6)]
const bool REUSE_BORDER = false;
[vk::constant_id(7)]
const bool USE_MOTION_VECTORS = true;
[vk::constant_id(8)]
const int GBUFFER_CHECK_MODE = GBUFFER_CHECK_ALWAYS;
// Take the motion vectors from in_mv instead of the gbuffer. For signals that do not live on the
// first opaque surface, such as a participating medium.
[vk::constant_id(9)]
const bool MV_OVERRIDE = false;

import merian_shaders.gbuffer;
import merian_shaders.utils.encoding;
import merian_shaders.utils.textures;
import merian_shaders.utils.interpolation;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.reprojection;
import merian_shaders.colors.colorspaces;

using merian;

// Depths are compared against this ceiling on both sides: f16 saturates to inf above it, and
// inf-vs-inf passes the relative depth test, which would validate geometry against far field.
static const float MAX_HISTORY_Z = 65504.0;

struct History {
    float3 normal;
    float linear_z;
    float length;
};

// history texel: .x = raw gbuffer enc_normal bits, .y = f16 linear_z | f16 length << 16
uint2 encode_history(uint enc_normal, float linear_z, float length) {
    return uint2(enc_normal, f32tof16(linear_z) | (f32tof16(length) << 16));
}

History decode_history(uint2 packed) {
    History history;
    history.normal = merian::decode_normal(packed.x);
    history.linear_z = f16tof32(packed.y & 0xffff);
    history.length = f16tof32(packed.y >> 16);
    return history;
}

// graph io
Sampler2D<float4> in_src;
ParameterBlock<GBuffer> in_gbuffer;
Sampler2D<float2> in_mv; // only read when MV_OVERRIDE
Sampler2D<float4> in_prev_out;
Sampler2D<uint2> in_prev_history;
RWTexture2D<float4> out_out;
RWTexture2D<uint2> out_history;

// percentile pass result, bound by the node
Sampler2D<float4> quartiles;

struct PushConstant {
    int firefly_filter_enable;
    float firefly_bias;
    float firefly_ipr_factor;
    float firefly_hard_clamp;

    float accum_alpha;
    float accum_max_hist;
    float normal_reject_cos;
    float z_reject_percent;
    int clear;

    float adaptive_alpha_reduction;
    float adaptive_alpha_ipr_factor;

    uint iteration;
};

[vk::push_constant]
ConstantBuffer<PushConstant> pc;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(uint3 ipos: SV_DispatchThreadID) {

    if (not_on_texture(ipos.xy, out_out))
        return;

    float4 src = in_src.Load(ipos); // contains second moment in a
    const uint enc_normal = in_gbuffer.get_encoded_normal(ipos.xy);
    const SurfaceData gbuf = in_gbuffer.get_surface(ipos.xy);
    const float linear_z = min(gbuf.linear_z, MAX_HISTORY_Z);

    float4 percentiles;
    if (pc.firefly_filter_enable != 0 ||
        (pc.adaptive_alpha_reduction > 0.0 && pc.clear != 1 && pc.accum_alpha != 0)) {
        const uint2 rounded_size = uint2(WG_ROUNDED_IRR_SIZE_X, WG_ROUNDED_IRR_SIZE_Y);
        percentiles = quartiles.SampleLevel((float2(ipos.xy) + 0.5) / rounded_size, 0);
    }

    // FIREFLY CLAMPING/SUPPRESSION
    {
        float max_l = pc.firefly_hard_clamp;
        if (pc.firefly_filter_enable != 0) {
            max_l = min(max_l, pc.firefly_bias + percentiles.y +
                                   pc.firefly_ipr_factor * (percentiles.y - percentiles.x));
        }

        // Scaling a non-finite sample would turn it into NaN (inf * (max_l / inf) == inf * 0), and
        // that NaN then spreads through the history and the denoiser.
        if (any(isnan(src)) || any(isinf(src))) {
            src = float4(0);
        } else {
            const float l = yuv_luminance(src.rgb);
            if (l > max_l) {
                const float factor = max_l / l;
                src.rgb *= factor;
                src.a *= factor * factor;
            }
        }
    }

    // CLEAR SHORTCUT
    if (pc.clear == 1 || pc.accum_alpha == 0) {
        out_out.Store(ipos.xy, src);
        out_history.Store(ipos.xy, encode_history(enc_normal, linear_z, 1.0));
        return;
    }

    // REPROJECT
    int2 prev_ipos;
    float max_history = pc.accum_max_hist;
    bool is_static; // the surface did not move on screen (motion vector ~0)
    RandomGenerator rng = XorShift32(pcg3d16(uint3(ipos.xy, pc.iteration)));
    if (USE_MOTION_VECTORS) {
        const float2 mv = MV_OVERRIDE ? in_mv[ipos.xy] : in_gbuffer.get_motion_vectors(ipos.xy);
        is_static = dot(mv, mv) < 1e-3 * 1e-3;
        float2 prev_pos = ipos.xy + mv;

        if (REUSE_BORDER) {
            // Attemp to reuse information at the image border.
            // This results in minor smearing but looks a lot better than
            // noise / SVGF blotches.
            if (reprojection_intersect_border(prev_pos, mv, texture_dimensions(out_out) - 1)) {
                // reset history to converge faster
                max_history = 2.0;
            }
        }

        if (FILTER_MODE == FILTER_MODE_STOCHASTIC_BILINEAR) {
            prev_ipos = reproject_pixel_stochastic(prev_pos, rng.next_float());
        } else {
            // FILTER_MODE == FILTER_MODE_NEAREST
            prev_ipos = int2(round(prev_pos));
        }
    } else {
        prev_ipos = ipos.xy;
        is_static = true;
        max_history = pc.accum_max_hist;
    }

    // GET PREVIOUS PIXEL
    float prev_history = 0.0;
    float4 prev_out = float4(0);

    const bool skip_gbuffer_check = GBUFFER_CHECK_MODE == GBUFFER_CHECK_NEVER ||
                                    (GBUFFER_CHECK_MODE == GBUFFER_CHECK_IF_MOVING && is_static);

    bool found = false;
    if (is_on_texture(prev_ipos, in_src)) {
        const History history = decode_history(in_prev_history[prev_ipos]);
        found = skip_gbuffer_check ||
                reprojection_valid(gbuf.normal, history.normal, pc.normal_reject_cos, linear_z,
                                   gbuf.delta_z, history.linear_z, pc.z_reject_percent);
        if (found) {
            prev_history = history.length;
            prev_out = in_prev_out[prev_ipos];
        }
    }

    if (EXTENDED_SEARCH && !skip_gbuffer_check) {
        if (!found) {
            float sum_w = 0.0;

            [[unroll]]
            for (int i = 0; i <= 9; i++) {
                const int2 offset = int2(round(rng.next_float2() * 4)) - 2;
                const int2 tc = prev_ipos + offset;

                if (not_on_texture(tc, in_src))
                    continue;

                const History history = decode_history(in_prev_history[tc]);

                const float w = reprojection_weight(
                    gbuf.normal, history.normal, pc.normal_reject_cos * 0.8,
                    linear_z, gbuf.delta_z, offset, gbuf.grad_z,
                    history.linear_z, pc.z_reject_percent * 2.);

                if (w > 0.0001) {
                    prev_history += w * history.length;
                    prev_out += w * in_prev_out[tc];
                    sum_w += w;
                }
            }

            if (sum_w > 0.001) {
                found = true;
                prev_out /= sum_w;
                prev_history /= sum_w;
            }
        }
    }

    if (found && !any(isnan(prev_out)) && !any(isinf(prev_out))) {

        if (pc.adaptive_alpha_reduction > 0.0) {
            const float ipr = pc.adaptive_alpha_ipr_factor * (percentiles.w - percentiles.z);
            const float prev_l = yuv_luminance(prev_out.rgb);
            float adaptive_alpha =
                (1.0 - pc.adaptive_alpha_reduction *
                           linearstep(percentiles.w, percentiles.w + ipr, prev_l));
            adaptive_alpha *=
                (1.0 - pc.adaptive_alpha_reduction *
                           (1.0 - linearstep(percentiles.z - ipr, percentiles.z, prev_l)));

            // update history length to match reduced alpha:
            max_history = min(1.0 / (1.0 - adaptive_alpha), max_history);
        }

        const float new_history = min(prev_history + 1, max_history);
        const float alpha = max(1. - pc.accum_alpha, 1. / new_history);

        const float4 result = lerp(prev_out, src, alpha);

        out_out.Store(ipos.xy, result);
        out_history.Store(ipos.xy, encode_history(enc_normal, linear_z, new_history));
    } else {
        out_out.Store(ipos.xy, src);
        out_history.Store(ipos.xy, encode_history(enc_normal, linear_z, 1.0));
    }
}
