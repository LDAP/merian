#include "merian-graph/nodes/exposure/layout.slangh"

import merian_shaders.utils.textures;
import merian_shaders.utils.camera;

using merian;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, out_out))
        return;

    const float exposure =
        min(max(max_luminance_to_exposure(out_avg_luminance[0]), params.min_exposure), params.max_exposure);

    float4 color = in_src[ipos];
    color.rgb *= exposure;

    out_out.Store(ipos, color);
}
