#include "merian-graph/nodes/median_approx/layout.slangh"

import merian_shaders.utils.textures;

using merian;

static const uint hist_size = workgroup_size_x * workgroup_size_y;
groupshared uint[hist_size] local_hist;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(uint local_index: SV_GroupIndex) {
    const uint local_bin = out_histogram[local_index];
    local_hist[local_index] = local_bin * local_index;
    GroupMemoryBarrierWithGroupSync();

    for (uint split = (hist_size >> 1); split > 0; split >>= 1) {
        if (local_index < split) {
            local_hist[local_index] += local_hist[local_index + split];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_index == 0) {
        const uint2 image_size = uint2(texture_dimensions(in_src));
        const float num_not_lowest = max(image_size.x * image_size.y - float(local_bin), 1.0);

        const float average_bin = (local_hist[0] / num_not_lowest) - 1.0;
        const float approx_median =
            exp2(((average_bin / float(hist_size - 2.)) * params.max) + params.min);

        out_median[0] = approx_median;
    }
}
