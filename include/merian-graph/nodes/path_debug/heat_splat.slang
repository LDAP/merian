import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.colors.colorspaces;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

void grid_add(const float3 pos, const float weight) {
    const int3 cell = int3(floor(pos / params.heat_cell_size));
    const uint slot = path_debug_grid_find(path_debug_cell_key(cell), true);
    if (slot == PATH_DEBUG_GRID_INVALID) {
        return;
    }
    const uint index = params.grid_slots + slot;
    uint expected = grid[index];
    for (;;) {
        uint orig;
        InterlockedCompareExchange(grid[index], expected, asuint(asfloat(expected) + weight),
                                   orig);
        if (orig == expected) {
            break;
        }
        expected = orig;
    }
}

// Splats recorded path vertices into the world-space heat grid.
[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint count = min(in_records[PATH_RECORD_HDR_PATH_COUNT],
                           path_record_path_capacity(in_records));
    const uint i = gid.x;
    if (i >= count) {
        return;
    }

    const PathRecordPath p = path_record_load_path(in_records, i);
    // follow the path filter, so the grid describes the same set as every other view
    if (p.vertex_count == 0 || !path_debug_matched(i, 0)) {
        return;
    }
    if (params.heat_scope == 1 && any(p.pixel != path_debug_selected_pixel())) {
        return;
    }

    const uint vro = path_record_vertex_region_offset(in_records);
    float downstream = 0.f;
    for (int k = int(p.vertex_count) - 1; k >= 0; k--) {
        const PathRecordVertex v = path_record_load_vertex(in_records, vro, p.vertex_offset + k);
        // radiance leaving this vertex towards the camera
        downstream += yuv_luminance(v.contrib);
        if (uint(k) < params.heat_bounce_min || uint(k) > params.heat_bounce_max) {
            continue;
        }
        // environment pseudo-vertices sit at a huge distance and would own the max
        if (distance(v.pos, camera.position) > 1e6f) {
            continue;
        }
        float weight;
        switch (params.heat_mode) {
        case PATH_DEBUG_HEAT_CONTRIBUTION:
            weight = downstream;
            break;
        case PATH_DEBUG_HEAT_THROUGHPUT:
            weight = v.terminal ? 0.f : v.throughput;
            break;
        default:
            weight = 1.f;
            break;
        }
        if (weight > 0.f) {
            grid_add(v.pos, weight);
        }
    }
}
