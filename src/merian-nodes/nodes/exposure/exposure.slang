#include "layout.slangh"

import merian_shaders.utils.textures;
import merian_shaders.utils.camera;

using merian;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, img_out))
        return;

    // Visualize histogram (disable luminance)
    // const uint hist_size = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    // const uint bin = ipos.x * (hist_size - 1) / imageSize(img_out).x;
    // imageStore(img_out, ipos, histogram[bin] / imageSize(img_out).x > (imageSize(img_out).y -
    // ipos.y - 1) ? float4(1) : float4(0)); return;

    const float exposure =
        min(max(max_luminance_to_exposure(luminance[0]), params.min_exposure), params.max_exposure);

    float4 color = img_src.Load(int3(ipos, 0));
    color.rgb *= exposure;

    img_out.Store(ipos, color);
}
