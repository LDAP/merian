import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

// Top-k and firefly modes: append every path at or above the threshold to the draw list.
[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    if (params.mode == PATH_DEBUG_MODE_SELECTED_PIXEL) {
        return;
    }
    const uint count = min(in_records[PATH_RECORD_HDR_PATH_COUNT],
                           path_record_path_capacity(in_records));
    const uint i = gid.x;
    if (i >= count) {
        return;
    }

    const PathRecordPath p = path_record_load_path(in_records, i);
    if (p.vertex_count == 0 || p.luminance < asfloat(state[PATH_DEBUG_STATE_THRESHOLD])) {
        return;
    }
    if (!path_debug_matched(i, 0)) {
        return;
    }

    uint slot;
    InterlockedAdd(state[PATH_DEBUG_STATE_DRAW_COUNT], 1, slot);
    if (slot < params.max_draw) {
        state[PATH_DEBUG_STATE_DRAW_LIST + slot] = i;
    }
}
