import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.colors.colorspaces;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

void grid_add(const float3 pos, const float weight) {
    const uint slot = path_debug_grid_find(path_debug_grid_key(pos), true);
    if (slot == PATH_DEBUG_GRID_INVALID) {
        return;
    }
    path_debug_atomic_add_float(grid, grid_f, params.grid_slots + slot, weight);
}

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint count = path_record_path_count(in_records);
    const uint i = gid.x;
    if (i >= count) {
        return;
    }

    const PathRecordPath p = path_record_load_path(in_records, i);
    if (p.vertex_count == 0 || !path_debug_matched(i)) {
        return;
    }

    const uint vertex_region = path_record_vertex_region_offset(in_records);
    uint chain = path_debug_chain_length(in_records, p, vertex_region);
    float downstream = 0.f;
    float nee_at_parent = 0.f;
    for (int k = int(p.vertex_count) - 1; k >= 0; k--) {
        const PathRecordVertex v =
            path_record_load_vertex(in_records, vertex_region, p.vertex_offset + k);
        const float lum = path_debug_finite_or_zero(yuv_luminance(v.contrib));
        if (v.nee) {
            nee_at_parent += lum;
            continue;
        }
        downstream += lum + nee_at_parent;
        nee_at_parent = 0.f;
        chain--;
        if (chain < params.heat_bounce_min || chain > params.heat_bounce_max) {
            continue;
        }
        if (distance(v.pos, camera.position) > PATH_DEBUG_ENV_DISTANCE) {
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
