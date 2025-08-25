#define FILTER_MODE_NEAREST 0
#define FILTER_MODE_STOCHASTIC_BILINEAR 1

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

#include "graph_layout.slangh"

[vk::binding(0, 1)]
Sampler2D<float4> img_quartiles;

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

import merian_shaders.utils.textures;
import merian_shaders.utils.interpolation;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.reprojection;
import merian_shaders.colors.colorspaces;

using merian;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(uint3 ipos: SV_DispatchThreadID) {

    if (not_on_texture(ipos.xy, img_out))
        return;

    float4 src = img_src.Load(ipos); // contains second moment in a

    float4 percentiles;
    if (pc.firefly_filter_enable != 0 ||
        (pc.adaptive_alpha_reduction > 0.0 && pc.clear != 1 && pc.accum_alpha != 0)) {
        const uint2 rounded_size = uint2(WG_ROUNDED_IRR_SIZE_X, WG_ROUNDED_IRR_SIZE_Y);
        percentiles = img_quartiles.SampleLevel((float2(ipos.xy) + 0.5) / rounded_size, 0);
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
        img_out.Store(ipos.xy, src);
        img_history.Store(ipos.xy, uint4(1, 0, 0, 0));
        return;
    }

    // REPROJECT
    int2 prev_ipos;
    float max_history = pc.accum_max_hist;
    RandomGenerator rng = XorShift32(pcg3d16(uint3(ipos.xy, pc.iteration)));
    if (USE_MOTION_VECTORS) {
        const float2 mv = img_mv.Load(ipos).xy;
        float2 prev_pos = ipos.xy + mv;

        if (REUSE_BORDER) {
            // Attemp to reuse information at the image border.
            // This results in minor smearing but looks a lot better than
            // noise / SVGF blotches.
            if (reprojection_intersect_border(prev_pos, mv, texture_dimensions(img_out) - 1)) {
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
        max_history = pc.accum_max_hist;
    }

    // GET CURRENT PIXEL INFO
    GBuffer gbuf = GBuffer::load(gbuffer, ipos.xy);

    // GET PREVIOUS PIXEL
    float prev_history = 0.0;
    float4 prev_out = float4(0);

    bool found = false;
    if (is_on_texture(prev_ipos, img_src)) {
        const CompressedGBuffer prev_gbuf = GBuffer::load_compressed(prev_gbuffer, prev_ipos);
        found = reprojection_valid(gbuf.get_normal(), prev_gbuf.decompress_normal(),
                                   pc.normal_reject_cos, gbuf.get_linear_z(), gbuf.get_delta_z(),
                                   prev_gbuf.decompress_linear_z(), pc.z_reject_percent);

        if (found) {
            prev_history = img_prev_history.Load(int3(prev_ipos, 0)).r;
            prev_out = img_prev_out.Load(int3(prev_ipos, 0));
        }
    }

    if (EXTENDED_SEARCH) {
        if (!found) {
            float sum_w = 0.0;

            [[unroll]]
            for (int i = 0; i <= 9; i++) {
                const int2 offset = int2(round(rng.next_float2() * 4)) - 2;
                const int2 tc = prev_ipos + offset;

                if (not_on_texture(tc, img_src))
                    continue;

                const CompressedGBuffer prev_gbuf = GBuffer::load_compressed(prev_gbuffer, tc);

                const float w = reprojection_weight(
                    gbuf.get_normal(), prev_gbuf.decompress_normal(), pc.normal_reject_cos * 0.8,
                    gbuf.get_linear_z(), gbuf.get_delta_z(), offset, gbuf.get_grad_z(),
                    prev_gbuf.decompress_linear_z(), pc.z_reject_percent * 2.);

                if (w > 0.0001) {
                    prev_history += w * img_prev_history.Load(uint3(tc, 0)).r;
                    prev_out += w * img_prev_out.Load(uint3(tc, 0));
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

        img_out.Store(ipos.xy, result);
        img_history.Store(ipos.xy, float4(new_history, 0, 0, 0));
    } else {
        img_out.Store(ipos.xy, src);
        img_history.Store(ipos.xy, float4(1, 0, 0, 0));
    }
}
