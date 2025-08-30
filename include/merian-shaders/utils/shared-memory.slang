#pragma once

namespace merian {

static uint3 __gl_WorkGroupID : SV_GroupID;
static uint __gl_LocalInvocationIndex : SV_GroupIndex;

// Computes indices to load the halo into shared memory.
//
// Assumes that the shared memory size is workgroupSize.xy + 2 * halo_radius.
//
// Returns true, if this invocation should load something into shared memory.
// This function requires that gl_WorkgroupSize.x == gl_WorkgroupSize.y
// workgroupSize.x * gl_WorkgroupSize.y >= 2 * halo_radius * workgroupSize.x + 2 * halo_radius *
// gl_WorkgroupSize.y + 4 * halo_radius * halo_radius.
bool load_halo_index(int halo_radius,
                     uint2 workgroupSize,
                     out int2 shared_index,
                     out int2 global_index) {
    // |TL| TC |TR|
    // |L | C  |R |
    // |BL| BC |BR|

    int2 local_tile_offset;
    int2 lid;

    const uint split_size = halo_radius * (halo_radius + workgroupSize.x);
    const uint split_size_half = split_size / halo_radius;

    if (__gl_LocalInvocationIndex < 1 * split_size) {
        // load top (TL, TC)
        const uint index = __gl_LocalInvocationIndex - 0 * split_size;
        lid = int2(index % split_size_half, index / split_size_half);
        local_tile_offset = int2(0);
    } else if (__gl_LocalInvocationIndex < 2 * split_size) {
        // load right (TR, R)
        const uint index = __gl_LocalInvocationIndex - 1 * split_size;
        lid = int2(index / split_size_half, index % split_size_half);
        local_tile_offset = int2(halo_radius + workgroupSize.x, 0);
    } else if (__gl_LocalInvocationIndex < 3 * split_size) {
        // load bottom (BR, BC)
        const uint index = __gl_LocalInvocationIndex - 2 * split_size;
        lid = int2(index % split_size_half, index / split_size_half);
        local_tile_offset = int2(halo_radius, halo_radius + workgroupSize.x);
    } else if (__gl_LocalInvocationIndex < 4 * split_size) {
        // load left (BL, L)
        const uint index = __gl_LocalInvocationIndex - 3 * split_size;
        lid = int2(index / split_size_half, index % split_size_half);
        local_tile_offset = int2(0, halo_radius);
    }

    const int2 global_tile_offset = int2(workgroupSize.xy * __gl_WorkGroupID.xy) - halo_radius;

    global_index = global_tile_offset + local_tile_offset + lid;
    shared_index = local_tile_offset + lid;
    return __gl_LocalInvocationIndex < 4 * split_size;
}

}
