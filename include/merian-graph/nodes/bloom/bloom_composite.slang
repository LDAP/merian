#include "merian-graph/nodes/bloom/layout.slangh"

import merian_shaders.utils.textures;

using merian;

Sampler2D<float4> in_src;
RWTexture2D<float4> out_out;
Sampler2D<float4> up_tex[MAX_LEVELS];

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, out_out))
        return;

    const float4 src = in_src[ipos];
    uint width, height;
    out_out.GetDimensions(width, height);
    const float3 bloom = tent(up_tex[0], pixel_uv(ipos, width, height), pc.radius);

    switch (MODE) {
    case 0: {
        out_out[ipos] = float4(lerp(src.rgb, bloom, pc.intensity), src.a);
        break;
    }
    case 1: {
        out_out[ipos] = float4(bloom, src.a);
        break;
    }
    case 2: {
        out_out[ipos] = src;
        break;
    }
    }
}
