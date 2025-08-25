#include "layout.slangh"

import merian_shaders.utils.textures;

using merian;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, img_out))
        return;

    const float weights[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };
    float3 bloom = img_interm.Load(ipos).rgb * weights[0];

    // blur horizontally (vertical pass in separate)
    for (int i = 1; i < 5; ++i) {
        bloom += img_interm.Load(clamp(ipos + int2(0, i), int2(0), texture_dimensions(img_src) - 1))
                     .rgb *
                 weights[i];
        bloom += img_interm.Load(clamp(ipos - int2(0, i), int2(0), texture_dimensions(img_src) - 1))
                     .rgb *
                 weights[i];
    }

    switch (MODE) {
    case 0: {
        img_out.Store(ipos, float4(img_src.Load(int3(ipos, 0)).rgb + pc.strength * bloom, 1));
        break;
    }
    case 1: {
        img_out.Store(ipos, float4(pc.strength * bloom, 1));
        break;
    }
    case 2: {
        img_out.Store(ipos, float4(img_src.Load(int3(ipos, 0)).rgb, 1));
        break;
    }
    }
}
