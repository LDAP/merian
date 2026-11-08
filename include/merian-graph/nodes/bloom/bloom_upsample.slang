#include "merian-graph/nodes/bloom/layout.slangh"

Sampler2D<float4> down_tex[MAX_LEVELS];
Sampler2D<float4> up_tex[MAX_LEVELS];
[format("rgba32f")]
RWTexture2D<float4> up_img[MAX_LEVELS];

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    uint width, height;
    up_img[pc.level].GetDimensions(width, height);
    if (any(ipos >= int2(width, height)))
        return;

    const float2 uv = pixel_uv(ipos, width, height);
    float3 coarse;
    if (pc.coarse_is_down != 0) {
        coarse = tent(down_tex[pc.level + 1], uv, pc.radius);
    } else {
        coarse = tent(up_tex[pc.level + 1], uv, pc.radius);
    }
    const float3 fine = down_tex[pc.level].Load(int3(ipos, 0)).rgb;

    up_img[pc.level][ipos] = float4(pc.coarse_weight * coarse + pc.fine_weight * fine, 1);
}
