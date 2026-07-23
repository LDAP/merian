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

import merian_shaders.gbuffer;
import merian_shaders.utils.textures;
import merian_shaders.utils.interpolation;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.reprojection;
import merian_shaders.colors.colorspaces;

using merian;

// graph io
Sampler2D<float4> in_src;
ParameterBlock<GBuffer> in_gbuffer;
Sampler2D<float4> in_prev_out;
ParameterBlock<GBuffer> in_prev_gbuffer;
Sampler2D<float> in_prev_history;
RWTexture2D<float4> out_out;
RWTexture2D<float> out_history;

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

        const float l = yuv_luminance(src.rgb);
        if (l > max_l) {
            const float factor = max_l / l;
            src.rgb *= factor;
            src.a *= factor * factor;
        }
    }

    // CLEAR SHORTCUT
    if (pc.clear == 1 || pc.accum_alpha == 0) {
        out_out.Store(ipos.xy, src);
        out_history.Store(ipos.xy, 1);
        return;
    }

    // REPROJECT
    int2 prev_ipos;
    float max_history = pc.accum_max_hist;
    bool is_static; // the surface did not move on screen (motion vector ~0)
    RandomGenerator rng = XorShift32(pcg3d16(uint3(ipos.xy, pc.iteration)));
    if (USE_MOTION_VECTORS) {
        const float2 mv = in_gbuffer.get_motion_vectors(ipos.xy);
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

    // GET CURRENT PIXEL INFO
    SurfaceData gbuf = in_gbuffer.get_surface(ipos.xy);

    // GET PREVIOUS PIXEL
    float prev_history = 0.0;
    float4 prev_out = float4(0);

    const bool skip_gbuffer_check = GBUFFER_CHECK_MODE == GBUFFER_CHECK_NEVER ||
                                    (GBUFFER_CHECK_MODE == GBUFFER_CHECK_IF_MOVING && is_static);

    bool found = false;
    if (is_on_texture(prev_ipos, in_src)) {
        if (skip_gbuffer_check) {
            found = true;
        } else {
            const SurfaceData prev_gbuf = in_prev_gbuffer.get_surface(prev_ipos);
            found = reprojection_valid(gbuf.normal, prev_gbuf.normal,
                                       pc.normal_reject_cos, gbuf.linear_z, gbuf.delta_z,
                                       prev_gbuf.linear_z, pc.z_reject_percent);
        }

        if (found) {
            prev_history = in_prev_history[prev_ipos].r;
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

                const SurfaceData prev_gbuf = in_prev_gbuffer.get_surface(tc);

                const float w = reprojection_weight(
                    gbuf.normal, prev_gbuf.normal, pc.normal_reject_cos * 0.8,
                    gbuf.linear_z, gbuf.delta_z, offset, gbuf.grad_z,
                    prev_gbuf.linear_z, pc.z_reject_percent * 2.);

                if (w > 0.0001) {
                    prev_history += w * in_prev_history[tc].r;
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
        out_history.Store(ipos.xy, new_history);
    } else {
        out_out.Store(ipos.xy, src);
        out_history.Store(ipos.xy, 1);
    }
}
