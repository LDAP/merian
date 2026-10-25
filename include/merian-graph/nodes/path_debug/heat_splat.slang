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
    path_debug_atomic_add_float(grid, grid_f, 2 * params.grid_slots + slot, 1.f);
}

// the throughput before the chain vertex k's draw, 1 at the camera
float throughput_before(const PathRecordPath p, const uint vertex_region, const int k) {
    for (int v = k - 1; v >= 0; v--) {
        const uint meta =
            path_record_load_vertex_meta(in_records, vertex_region, p.vertex_offset + v);
        if (!path_debug_is_connection(meta)) {
            return path_record_load_vertex(in_records, vertex_region, p.vertex_offset + v)
                .throughput;
        }
    }
    return 1.f;
}

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint count = path_record_path_count(in_records);
    const uint i = gid.x;
    if (i >= count) {
        return;
    }

    // every path, or what the region and the filter hold of it
    const bool with_filter = params.false_color_filtered != 0;
    uint2 selected = uint2(0xFFFFFFFFu);
    if (with_filter) {
        selected = path_debug_path_events(i, true);
        if (all(selected == 0) && !path_debug_selection_open(true)) {
            return;
        }
    }
    const PathRecordPath p = path_debug_load_record(i);

    const uint vertex_region = path_record_vertex_region_offset(in_records);
    uint chain = path_debug_chain_length(in_records, p, vertex_region);
    float downstream = 0.f;
    float nee_at_parent = 0.f;
    uint delivered = 0;
    uint failed = 0;
    for (int k = int(p.vertex_count) - 1; k >= 0; k--) {
        const PathRecordVertex v =
            path_record_load_vertex(in_records, vertex_region, p.vertex_offset + k);
        const bool event = with_filter ? path_debug_is_event_vertex(selected, uint(k))
                                       : !path_debug_is_empty_draw(v);
        const float lum = event ? path_debug_finite_or_zero(yuv_luminance(v.contrib)) : 0.f;
        if (v.nee) {
            nee_at_parent += lum;
            delivered += event && lum > 0.f ? 1 : 0;
            failed += event && !(lum > 0.f) ? 1 : 0;
            continue;
        }
        const float arriving = downstream;
        downstream += lum + nee_at_parent;
        const uint connections_delivered = delivered;
        const uint connections_failed = failed;
        nee_at_parent = 0.f;
        delivered = 0;
        failed = 0;
        chain--;
        // path ends count the terminal vertex, an event or not
        const bool counted = path_debug_event_drawn(selected, uint(k), false, with_filter) ||
                             (params.heat_mode == PATH_DEBUG_HEAT_PATH_ENDS && v.terminal);
        if (!counted || chain < params.heat_bounce_min || chain > params.heat_bounce_max) {
            continue;
        }
        if (distance(v.pos, camera.position) > PATH_DEBUG_ENV_DISTANCE) {
            continue;
        }
        // averaged quantities count every vertex that has one
        float weight = 0.f;
        bool has = false;
        switch (params.heat_mode) {
        case PATH_DEBUG_HEAT_CONTRIBUTION:
            weight = downstream;
            break;
        case PATH_DEBUG_HEAT_THROUGHPUT:
            weight = v.throughput;
            has = !v.terminal;
            break;
        case PATH_DEBUG_HEAT_INCIDENT:
            weight = v.throughput > 0.f ? arriving / v.throughput : 0.f;
            has = !v.terminal && v.throughput > 0.f;
            break;
        case PATH_DEBUG_HEAT_SAMPLE_WEIGHT: {
            const float before = throughput_before(p, vertex_region, k);
            weight = before > 0.f ? v.throughput / before : 0.f;
            has = !v.terminal && before > 0.f;
            break;
        }
        case PATH_DEBUG_HEAT_NEE_SUCCESS:
            weight = float(connections_delivered);
            break;
        case PATH_DEBUG_HEAT_NEE_FAILED:
            weight = float(connections_failed);
            break;
        case PATH_DEBUG_HEAT_PATH_ENDS:
            weight = v.terminal ? 1.f : 0.f;
            break;
        default:
            weight = 1.f;
            break;
        }
        if (weight > 0.f || has) {
            grid_add(v.pos, weight);
        }
    }
}
