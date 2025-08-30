#pragma once

namespace merian {

// Computes indices to load the halo into shared memory.
//
// Assumes that the shared memory size is group_size.xy + 2 * halo_radius.
//
// Returns true, if this invocation should load something into shared memory.
// This function requires that group_size.x == group_size.y
// group_size.x * group_size.y >= 2 * halo_radius * group_size.x + 2 * halo_radius *
// group_size.y + 4 * halo_radius * halo_radius.
bool load_halo_index(const int halo_radius,
                     const uint2 group_size,
                     const uint2 group_id,
                     const uint group_index,
                     out int2 shared_index,
                     out int2 global_index) {
    // |TL| TC |TR|
    // |L | C  |R |
    // |BL| BC |BR|

    int2 local_tile_offset;
    int2 lid;

    const uint split_size = halo_radius * (halo_radius + group_size.x);
    const uint split_size_half = split_size / halo_radius;

    if (group_index < 1 * split_size) {
        // load top (TL, TC)
        const uint index = group_index - 0 * split_size;
        lid = int2(index % split_size_half, index / split_size_half);
        local_tile_offset = int2(0);
    } else if (group_index < 2 * split_size) {
        // load right (TR, R)
        const uint index = group_index - 1 * split_size;
        lid = int2(index / split_size_half, index % split_size_half);
        local_tile_offset = int2(halo_radius + group_size.x, 0);
    } else if (group_index < 3 * split_size) {
        // load bottom (BR, BC)
        const uint index = group_index - 2 * split_size;
        lid = int2(index % split_size_half, index / split_size_half);
        local_tile_offset = int2(halo_radius, halo_radius + group_size.x);
    } else if (group_index < 4 * split_size) {
        // load left (BL, L)
        const uint index = group_index - 3 * split_size;
        lid = int2(index / split_size_half, index % split_size_half);
        local_tile_offset = int2(0, halo_radius);
    }

    const int2 global_tile_offset = int2(group_size.xy * group_id) - halo_radius;

    global_index = global_tile_offset + local_tile_offset + lid;
    shared_index = local_tile_offset + lid;
    return group_index < 4 * split_size;
}

}
