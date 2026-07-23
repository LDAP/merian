#include "merian-graph/nodes/taa/config.h"

import merian_shaders.utils.textures;

using merian;

// Good blog post on TAA: https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail/

[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;

[vk::constant_id(2)]
const int INVERSE_MOTION = 0;

Sampler2D<float4> in_src;
Sampler2D<float4> in_prev_src;
Sampler2D<float4> in_mv; // in pixels

RWTexture2D<float4> out_out;

struct PushConstant {
    // higher value means more temporal reuse
    float temporal_alpha;
    int clamp_method;
    int enable_mv;
}

[vk::push_constant]
ConstantBuffer<PushConstant> params;

float4 merge_frames(const float4 current_color,
                    float4 previous_color,
                    const float alpha,
                    const int2 pixel) {
    switch (params.clamp_method) {
    case MERIAN_NODES_TAA_CLAMP_NONE:
        break;
    case MERIAN_NODES_TAA_CLAMP_MIN_MAX:
        float4 neigh_clamp_min_color = float4(1. / 0.);
        float4 neigh_clamp_max_color = float4(-1. / 0.);
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                const float4 color = texelFetchClamp(pixel + int2(j, i), in_src);
                neigh_clamp_min_color = min(neigh_clamp_min_color, color);
                neigh_clamp_max_color = max(neigh_clamp_max_color, color);
            }
        }
        previous_color = clamp(previous_color, neigh_clamp_min_color, neigh_clamp_max_color);
        break;
    case 2:
        float4 m1 = float4(0.0f);
        float4 m2 = float4(0.0f);
        const int r = 1;
        for (int yy = -r; yy <= r; yy++)
            for (int xx = -r; xx <= r; xx++) {
                const float4 b = texelFetchClamp(pixel + int2(xx, yy), in_src);
                m1 += b;
                m2 += b * b;
            }
        m1 /= (2.0 * r + 1) * (2.0 * r + 1);
        m2 /= (2.0 * r + 1) * (2.0 * r + 1);
        const float4 sigma = sqrt(max(float4(0), m2 - m1 * m1));
        const float thresh = 1.0; // params.rejection_threshold;
        previous_color =
            clamp(previous_color, max(float4(0), m1 - thresh * sigma), m1 + thresh * sigma);
        break;
    default:
        return float4(1, 0, 1, 1);
    }

    return lerp(current_color, previous_color, alpha);
}

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(const int2 pixel: SV_DispatchThreadID) {
    const int2 resolution = texture_dimensions(out_out);
    if (not_on_texture(pixel, out_out))
        return;

    float2 motion_vector;
    if (params.enable_mv != 0) {
        motion_vector = sample_motion_vector(in_mv, pixel, 1);
    } else {
        motion_vector = float2(0);
    }

    if (INVERSE_MOTION > 0) {
        motion_vector *= -1;
    }

    const float4 current_color = in_src[pixel];
    float4 previous_color =
        catmull_rom(in_prev_src, (pixel + motion_vector + .5) / float2(resolution));

    // previous texture is never initialized and might contain NaN (prevent poisoning)
    if (any(isnan(previous_color)) || any(isinf(previous_color)))
        previous_color = current_color;

    const float4 merged = merge_frames(current_color, previous_color, params.temporal_alpha, pixel);
    out_out.Store(pixel, merged);
}
