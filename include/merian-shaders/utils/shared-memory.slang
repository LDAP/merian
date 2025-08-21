#pragma once

namespace merian {

static uint3 __gl_WorkGroupID : SV_GroupID;
static uint __gl_LocalInvocationIndex : SV_GroupIndex;

bool load_halo_index(int halo_radius,
                     uint2 workgroupSize,
                     out int2 shared_index,
                     out int2 global_index) {
    uint split_size = uint(halo_radius) * (uint(halo_radius) + workgroupSize.x);
    uint split_size_half = split_size / uint(halo_radius);
    int2 lid;
    int2 local_tile_offset;
    if (__gl_LocalInvocationIndex < (1u * split_size)) {
        uint index = __gl_LocalInvocationIndex - (0u * split_size);
        lid = int2(int(index % split_size_half), int(index / split_size_half));
        local_tile_offset = int2(0, 0);
    } else {
        if (__gl_LocalInvocationIndex < (2u * split_size)) {
            uint index_1 = __gl_LocalInvocationIndex - (1u * split_size);
            lid = int2(int(index_1 / split_size_half), int(index_1 % split_size_half));
            local_tile_offset = int2(int(uint(halo_radius) + workgroupSize.x), 0);
        } else {
            if (__gl_LocalInvocationIndex < (3u * split_size)) {
                uint index_2 = __gl_LocalInvocationIndex - (2u * split_size);
                lid = int2(int(index_2 % split_size_half), int(index_2 / split_size_half));
                local_tile_offset = int2(halo_radius, int(uint(halo_radius) + workgroupSize.x));
            } else {
                if (__gl_LocalInvocationIndex < (4u * split_size)) {
                    uint index_3 = __gl_LocalInvocationIndex - (3u * split_size);
                    lid = int2(int(index_3 / split_size_half), int(index_3 % split_size_half));
                    local_tile_offset = int2(0, halo_radius);
                }
            }
        }
    }
    int2 global_tile_offset = int2(workgroupSize * __gl_WorkGroupID.xy) - halo_radius.xx;
    global_index = (global_tile_offset + local_tile_offset) + lid;
    shared_index = local_tile_offset + lid;
    return __gl_LocalInvocationIndex < (4u * split_size);
}

}
