#include "merian-graph/nodes/mean/layout.slangh"

import merian_shaders.utils.textures;

using merian;

// upper bound: one entry per lane would mean subgroup size 1
groupshared float4[workgroup_size_x * workgroup_size_y] subgroup_sums;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID, uint local_index: SV_GroupIndex, uint2 group_id: SV_GroupID) {
    float4 v;
    if (!is_on_texture(ipos, in_src)) {
        v = float4(0);
    } else {
        v = in_src[ipos] / params.divisor;
    }

    v = WaveActiveSum(v);

    const uint subgroup_id = local_index / WaveGetLaneCount();
    if (WaveIsFirstLane()) {
        subgroup_sums[subgroup_id] = v;
    }

    GroupMemoryBarrierWithGroupSync();

    if (local_index == 0) {
        const uint num_subgroups =
            (workgroup_size_x * workgroup_size_y + WaveGetLaneCount() - 1) / WaveGetLaneCount();
        float4 sum = float4(0);
        for (uint i = 0; i < num_subgroups; i++) {
            sum += subgroup_sums[i];
        }

        const int2 tex_size = texture_dimensions(in_src);
        const uint num_groups_y = (tex_size.y + workgroup_size_y - 1) / workgroup_size_y;
        out_mean[group_id.y + num_groups_y * group_id.x] = sum;
    }
}
