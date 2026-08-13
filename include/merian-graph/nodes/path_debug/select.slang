import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

// Per-path predicate; top-k and firefly modes bin luminances for the threshold pass,
// selected-pixel mode appends to the draw list directly.
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
    // the records may come from a renderer of a different size than the image this node composes
    if (p.vertex_count == 0 || any(p.pixel >= path_debug_dim())) {
        return;
    }

    const uint m = 6 * (p.pixel.y * params.dim_x + p.pixel.x);
    if (params.accumulate_moments != 0) {
        path_debug_atomic_add_float(moments, m + 0, p.luminance);
        path_debug_atomic_add_float(moments, m + 1, p.luminance * p.luminance);
        InterlockedAdd(moments[m + 2], 1);
    }

    const uint vro = path_record_vertex_region_offset(in_records);
    const uint scatter = path_debug_scatter_count(p, vro);
    if (!path_debug_filter_constraints(p, vro, scatter)) {
        return;
    }
    if (path_debug_expression_matches(1, p, vro, scatter)) {
        path_debug_set_match(i, 1);
    }
    if (!path_debug_expression_matches(0, p, vro, scatter)) {
        return;
    }
    path_debug_set_match(i, 0);

    if (params.accumulate_moments != 0) {
        path_debug_atomic_add_float(moments, m + 3, p.luminance);
        path_debug_atomic_add_float(moments, m + 4, p.luminance * p.luminance);
        InterlockedAdd(moments[m + 5], 1);
    }

    if (params.mode == PATH_DEBUG_MODE_SELECTED_PIXEL) {
        if (any(p.pixel != path_debug_selected_pixel())) {
            return;
        }
        InterlockedAdd(state[PATH_DEBUG_STATE_SELECTED], 1);
        uint slot;
        InterlockedAdd(state[PATH_DEBUG_STATE_DRAW_COUNT], 1, slot);
        if (slot < params.max_draw) {
            state[PATH_DEBUG_STATE_DRAW_LIST + slot] = i;
        }
        return;
    }

    InterlockedAdd(state[PATH_DEBUG_STATE_SELECTED], 1);
    InterlockedAdd(state[PATH_DEBUG_STATE_HIST + path_debug_lum_bin(p.luminance)], 1);
}
