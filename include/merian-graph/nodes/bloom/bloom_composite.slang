#include "merian-graph/nodes/bloom/layout.slangh"

import merian_shaders.utils.textures;

using merian;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, out_out))
        return;

    const float weights[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };
    float3 bloom = out_interm.Load(ipos).rgb * weights[0];

    // blur vertically (horizontal pass in separate)
    for (int i = 1; i < 5; ++i) {
        bloom += out_interm.Load(clamp(ipos + int2(0, i), int2(0), texture_dimensions(in_src) - 1))
                     .rgb *
                 weights[i];
        bloom += out_interm.Load(clamp(ipos - int2(0, i), int2(0), texture_dimensions(in_src) - 1))
                     .rgb *
                 weights[i];
    }

    switch (MODE) {
    case 0: {
        out_out.Store(ipos, float4(in_src[ipos].rgb + pc.strength * bloom, 1));
        break;
    }
    case 1: {
        out_out.Store(ipos, float4(pc.strength * bloom, 1));
        break;
    }
    case 2: {
        out_out.Store(ipos, float4(in_src[ipos].rgb, 1));
        break;
    }
    }
}
