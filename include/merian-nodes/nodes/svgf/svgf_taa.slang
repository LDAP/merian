
#include "svgf_graph_layout.slangh"

import merian_shaders.utils.textures;

[vk_binding(0, 1)]
Sampler2D<float4> img_filter_result;
[vk_binding(1, 1)]
Sampler2D<float4> img_filter_unused; // unused needed to match the filter layout
[vk_binding(2, 1)]
Sampler2D<uint4> img_gbuf_result;
[vk_binding(3, 1)]
Sampler2D<uint4> img_gbuf_unused; // unused needed to match the filter layout

[vk::constant_id(2)]
const int DEBUG = 0;
[vk::constant_id(3)]
const int FILTER_PREV = 0;
[vk::constant_id(4)]
const int CLAMPING = 0;
[vk::constant_id(5)]
const int MV_SAMPLING = 0;
[vk::constant_id(6)]
const bool USE_MOTION_VECTORS = true;

struct PushConstant {
    float blend_alpha;
    float rejection_threshold;
};

[vk::push_constant]
ConstantBuffer<PushConstant> params;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(const int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, img_out))
        return;

    float2 mv;
    if (USE_MOTION_VECTORS) {
        if (MV_SAMPLING == 0)
            mv = img_mv.Load(uint3(ipos, 0)).rg;
        else if (MV_SAMPLING == 1)
            mv = sample_motion_vector(img_mv, ipos, 1);
    } else {
        mv = float2(0);
    }

    const float4 filter_result = img_filter_result.Load(uint3(ipos, 0));

    if (DEBUG == 1) {
        img_out.Store(ipos, float4(filter_result.rgb, 1));
        return;
    }
    if (DEBUG == 2) {
        img_out.Store(ipos, float4(filter_result.a / 256));
        return;
    }
    if (DEBUG == 3) {
        const float3 normal = GBuffer.load_and_decompress_normal(gbuffer, ipos);
        img_out.Store(ipos, float4(normal, 1));
        return;
    }
    if (DEBUG == 4) {
        img_out.Store(ipos, float4(GBuffer.load_and_decompress_linear_z(gbuffer, ipos) / 1000));
        return;
    }
    if (DEBUG == 5) {
        img_out.Store(ipos, img_albedo.Load(uint3(ipos, 0)));
        return;
    }
    if (DEBUG == 6) {
        img_out.Store(ipos, float4(GBuffer.load_and_decompress_grad_z(gbuffer, ipos), 0, 0));
        return;
    }
    if (DEBUG == 7) {
        img_out.Store(ipos, float4(float(any(isnan(filter_result.rgb))),
                                   float(any(isinf(filter_result.rgb))), 0, 0));
        return;
    }
    if (DEBUG == 8) {
        const float2 size = texture_dimensions(img_mv, 0);
        img_out.Store(ipos,
                      float4(abs(img_mv.Load(uint3(ipos, 0)).rg) / max(size.x, size.y), 0, 0));
        return;
    }

    const float3 irr = filter_result.rgb;
    const float3 albedo = img_albedo.Load(uint3(ipos, 0)).rgb; // multiply albedo

    float3 prev_out = float3(0);
    if (FILTER_PREV == 0) {
        prev_out = img_prev_out.Load(uint3(int2(round(ipos + mv)), 0)).rgb;
    } else if (FILTER_PREV == 1) {
        prev_out =
            catmull_rom(img_prev_out, (ipos + mv + 0.5) / float2(texture_dimensions(img_out))).rgb;
    }

    if (any(isnan(prev_out)) || any(isinf(prev_out))) {
        prev_out = irr * albedo;
    }

    if (CLAMPING == 0) {
        // min max clamping
        float3 neigh_clamp_min_color = float3(1. / 0.);
        float3 neigh_clamp_max_color = float3(-1. / 0.);
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                const float3 a = img_albedo.Load(uint3(ipos + int2(i, j), 0)).rgb;
                const float3 irr = img_filter_result.Load(uint3(ipos + int2(i, j), 0)).rgb;
                const float3 color = (a * irr);
                neigh_clamp_min_color = min(neigh_clamp_min_color, color);
                neigh_clamp_max_color = max(neigh_clamp_max_color, color);
            }
        }
        prev_out = clamp(prev_out, neigh_clamp_min_color, neigh_clamp_max_color);
    } else if (CLAMPING == 1) {
        // moments clamping
        float3 m1 = float3(0.0f);
        float3 m2 = float3(0.0f);
        const int r = 1;
        for (int yy = -r; yy <= r; yy++)
            for (int xx = -r; xx <= r; xx++) {
                const float3 a = img_albedo.Load(uint3(ipos + int2(xx, yy), 0)).rgb;
                const float3 irr = img_filter_result.Load(uint3(ipos + int2(xx, yy), 0)).rgb;
                const float3 b = (a * irr);
                m1 += b;
                m2 += b * b;
            }
        m1 /= (2.0 * r + 1) * (2.0 * r + 1);
        m2 /= (2.0 * r + 1) * (2.0 * r + 1);

        const float3 sigma = sqrt(max(float3(0), m2 - m1 * m1));
        const float thresh = params.rejection_threshold;
        prev_out = clamp(prev_out, max(float3(0), m1 - thresh * sigma), m1 + thresh * sigma);
    }

    const float3 beauty = irr * albedo;
    const float3 blended = lerp(beauty, prev_out, params.blend_alpha);

    img_out.Store(ipos, float4(blended, filter_result.a));
}

