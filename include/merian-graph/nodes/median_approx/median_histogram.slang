#include "merian-graph/nodes/median_approx/layout.slangh"

import merian_shaders.utils.textures;

using merian;

static const uint hist_size = workgroup_size_x * workgroup_size_y;
groupshared uint[hist_size] local_hist;

uint get_bin(const float value) {
    if (value < params.min) {
        return 0;
    }

    const float normalized = (value - params.min) / params.max;
    const uint bin = uint(normalized * (hist_size - 2) + 1);

    return clamp(bin, 0, hist_size - 1);
}

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID, uint local_index: SV_GroupIndex) {
    local_hist[local_index] = 0;
    GroupMemoryBarrierWithGroupSync();

    if (is_on_texture(ipos, in_src)) {
        float value;
        if (COMPONENT == 0) {
            value = in_src[ipos].r;
        } else if (COMPONENT == 1) {
            value = in_src[ipos].g;
        } else if (COMPONENT == 2) {
            value = in_src[ipos].b;
        } else {
            value = in_src[ipos].a;
        }

        const uint bin = get_bin(value);
        InterlockedAdd(local_hist[bin], 1);
    }
    GroupMemoryBarrierWithGroupSync();

    InterlockedAdd(out_histogram[local_index], local_hist[local_index]);
}
