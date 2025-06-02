#ifndef _MERIAN_SHADERS_SHARED_MEMORY_H_
#define _MERIAN_SHADERS_SHARED_MEMORY_H_

#extension GL_KHR_shader_subgroup_basic : enable

// Computes indices to load the halo into shared memory.
//
// Assumes that the shared memory size is workgroupSize.xy + 2 * halo_radius.
// 
// Returns true, if this invocation should load something into shared memory.
// This function requires that gl_WorkgroupSize.x == gl_WorkgroupSize.y
// workgroupSize.x * gl_WorkgroupSize.y >= 2 * halo_radius * workgroupSize.x + 2 * halo_radius * gl_WorkgroupSize.y + 4 * halo_radius * halo_radius.
bool load_halo_index(const int halo_radius, const uvec2 workgroupSize, out ivec2 shared_index, out ivec2 global_index) {
    // |TL| TC |TR|
    // |L | C  |R |
    // |BL| BC |BR|
    
    ivec2 local_tile_offset;
    ivec2 lid;
    
    const uint split_size = halo_radius * (halo_radius + workgroupSize.x);
    const uint split_size_half = split_size / halo_radius;
    
    if (gl_LocalInvocationIndex < 1 * split_size) {
        // load top (TL, TC)
        const uint index = gl_LocalInvocationIndex - 0 * split_size;
        lid = ivec2(index % split_size_half, index / split_size_half);
        local_tile_offset = ivec2(0);
    } else if (gl_LocalInvocationIndex < 2 * split_size) {
        // load right (TR, R) 
        const uint index = gl_LocalInvocationIndex - 1 * split_size;
        lid = ivec2(index / split_size_half, index % split_size_half);
        local_tile_offset = ivec2(halo_radius + workgroupSize.x, 0);
    } else if (gl_LocalInvocationIndex < 3 * split_size) {
        // load bottom (BR, BC)
        const uint index = gl_LocalInvocationIndex - 2 * split_size;
        lid = ivec2(index % split_size_half, index / split_size_half);
        local_tile_offset = ivec2(halo_radius, halo_radius + workgroupSize.x);
    } else if (gl_LocalInvocationIndex < 4 * split_size) {
        // load left (BL, L)
        const uint index = gl_LocalInvocationIndex - 3 * split_size;
        lid = ivec2(index / split_size_half, index % split_size_half);
        local_tile_offset = ivec2(0, halo_radius);
    }

    const ivec2 global_tile_offset = ivec2(workgroupSize.xy * gl_WorkGroupID.xy) - halo_radius;

    global_index = global_tile_offset + local_tile_offset + lid;
    shared_index = local_tile_offset + lid;
    return gl_LocalInvocationIndex < 4 * split_size;
}

#endif // _MERIAN_SHADERS_SHARED_MEMO_H_
