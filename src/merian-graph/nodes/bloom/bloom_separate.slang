#include "layout.slangh"

import merian_shaders.utils.textures;
import merian_shaders.colors.colorspaces;

using merian;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, img_out))
        return;

    const float weights[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };
    float3 result = float3(0);
    const float3 c = img_src.Load(int3(ipos, 0)).rgb;
    if (yuv_luminance(c) > pc.threshold) {
        result += c * weights[0];
    }

    // blur horizontally (vertical pass in composite)
    for (int i = 1; i < 5; ++i) {
        const float3 r =
            img_src
                .Load(int3(clamp(ipos + int2(i, 0), int2(0), texture_dimensions(img_src) - 1), 0))
                .rgb;
        const float3 l =
            img_src
                .Load(int3(clamp(ipos - int2(i, 0), int2(0), texture_dimensions(img_src) - 1), 0))
                .rgb;

        if (yuv_luminance(r) > pc.threshold)
            result += r * weights[i];
        if (yuv_luminance(l) > pc.threshold)
            result += l * weights[i];
    }

    img_interm.Store(ipos, float4(result, 1));
}
