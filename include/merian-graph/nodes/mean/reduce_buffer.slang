#include "merian-graph/nodes/mean/layout.slangh"


// upper bound: one entry per lane would mean subgroup size 1
groupshared float4[workgroup_size_x * workgroup_size_y] subgroup_sums;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(uint local_index: SV_GroupIndex, uint2 group_id: SV_GroupID) {
    const uint index = params.offset * (local_index + group_id.x * workgroup_size_x);

    float4 v;
    if (index < params.size) {
        v = out_mean[index];
    } else {
        v = float4(0);
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

        out_mean[index] = sum;
    }
}
