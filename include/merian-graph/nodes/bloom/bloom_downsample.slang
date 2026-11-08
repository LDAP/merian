#include "merian-graph/nodes/bloom/layout.slangh"

import merian_shaders.colors.colorspaces;

using merian;

Sampler2D<float4> in_src;
Sampler2D<float4> down_tex[MAX_LEVELS];
[format("rgba32f")]
RWTexture2D<float4> down_img[MAX_LEVELS];

float3 fetch(const float2 uv) {
    if (pc.level == 0) {
        const float3 color = in_src.SampleLevel(uv, 0).rgb;
        return all(isfinite(color)) ? color : float3(0);
    }
    return down_tex[pc.level - 1].SampleLevel(uv, 0).rgb;
}

float2 source_texel() {
    uint width, height;
    if (pc.level == 0) {
        in_src.GetDimensions(width, height);
    } else {
        down_tex[pc.level - 1].GetDimensions(width, height);
    }
    return 1.0 / float2(width, height);
}

float karis_weight(const float3 box) {
    return pc.karis != 0 ? 1.0 / (1.0 + yuv_luminance(box)) : 1.0;
}

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    uint width, height;
    down_img[pc.level].GetDimensions(width, height);
    if (any(ipos >= int2(width, height)))
        return;

    const float2 t = source_texel();
    const float2 uv = pixel_uv(ipos, width, height);

    const float3 a = fetch(uv + t * float2(-2, -2));
    const float3 b = fetch(uv + t * float2(0, -2));
    const float3 c = fetch(uv + t * float2(2, -2));
    const float3 d = fetch(uv + t * float2(-2, 0));
    const float3 e = fetch(uv);
    const float3 f = fetch(uv + t * float2(2, 0));
    const float3 g = fetch(uv + t * float2(-2, 2));
    const float3 h = fetch(uv + t * float2(0, 2));
    const float3 i = fetch(uv + t * float2(2, 2));
    const float3 j = fetch(uv + t * float2(-1, -1));
    const float3 k = fetch(uv + t * float2(1, -1));
    const float3 l = fetch(uv + t * float2(-1, 1));
    const float3 m = fetch(uv + t * float2(1, 1));

    const float3 boxes[5] = { (j + k + l + m) * 0.25, (a + b + d + e) * 0.25,
                              (b + c + e + f) * 0.25, (d + e + g + h) * 0.25,
                              (e + f + h + i) * 0.25 };
    const float box_weights[5] = { 0.5, 0.125, 0.125, 0.125, 0.125 };

    float3 sum = float3(0);
    float total = 0;
    for (int n = 0; n < 5; n++) {
        const float w = box_weights[n] * karis_weight(boxes[n]);
        sum += boxes[n] * w;
        total += w;
    }

    down_img[pc.level][ipos] = float4(sum / total, 1);
}
