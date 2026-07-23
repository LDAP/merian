
[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;

import merian_shaders.gbuffer;
import merian_shaders.utils.textures;

using merian;

// graph-bound
ParameterBlock<GBuffer> in_gbuffer;
Sampler2D<float4> in_prev_out;
RWTexture2D<float4> out_out;

// internal: final filter iteration result (bound by the node)
Sampler2D<float4> img_filter_result;

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
[vk::constant_id(7)]
const bool MODULATE_ALBEDO = true;

struct PushConstant {
    float blend_alpha;
    float rejection_threshold;
};

[vk::push_constant]
ConstantBuffer<PushConstant> params;

float3 modulation_albedo(uint2 pixel) {
    return MODULATE_ALBEDO ? in_gbuffer.get_demodulation_albedo(pixel) : float3(1);
}

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(const int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, out_out))
        return;

    float2 mv;
    if (USE_MOTION_VECTORS) {
        if (MV_SAMPLING == 0)
            mv = in_gbuffer.get_motion_vectors(uint2(ipos));
        else if (MV_SAMPLING == 1)
            mv = in_gbuffer.get_motion_vectors_dilated(ipos, 1);
    } else {
        mv = float2(0);
    }

    const float4 filter_result = img_filter_result[ipos];

    if (DEBUG == 1) {
        out_out.Store(ipos, float4(filter_result.rgb, 1));
        return;
    }
    if (DEBUG == 2) {
        out_out.Store(ipos, float4(filter_result.a / 256));
        return;
    }
    if (DEBUG == 3) {
        const float3 normal = in_gbuffer.get_normal(uint2(ipos));
        out_out.Store(ipos, float4(normal, 1));
        return;
    }
    if (DEBUG == 4) {
        out_out.Store(ipos, float4(in_gbuffer.get_linear_z(uint2(ipos)) / 1000));
        return;
    }
    if (DEBUG == 5) {
        out_out.Store(ipos, float4(in_gbuffer.get_albedo(uint2(ipos)).reflection, 1));
        return;
    }
    if (DEBUG == 6) {
        out_out.Store(ipos, float4(float2(in_gbuffer.get_grad_z(uint2(ipos))), 0, 0));
        return;
    }
    if (DEBUG == 7) {
        out_out.Store(ipos, float4(float(any(isnan(filter_result.rgb))),
                                   float(any(isinf(filter_result.rgb))), 0, 0));
        return;
    }
    if (DEBUG == 8) {
        const float2 size = texture_dimensions(in_gbuffer.tex2, 0);
        out_out.Store(
            ipos,
            float4(abs(in_gbuffer.get_motion_vectors(uint2(ipos))) / max(size.x, size.y), 0, 0));
        return;
    }

    const float3 irr = filter_result.rgb;
    const float3 albedo = modulation_albedo(uint2(ipos));

    float3 prev_out = float3(0);
    if (FILTER_PREV == 0) {
        prev_out = in_prev_out.Load(uint3(int2(round(ipos + mv)), 0)).rgb;
    } else if (FILTER_PREV == 1) {
        prev_out =
            catmull_rom(in_prev_out, (ipos + mv + 0.5) / float2(texture_dimensions(out_out))).rgb;
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
                const float3 a = modulation_albedo(uint2(ipos + int2(i, j)));
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
                const float3 a = modulation_albedo(uint2(ipos + int2(xx, yy)));
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

    out_out.Store(ipos, float4(blended, filter_result.a));
}
