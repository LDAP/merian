import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

void collect_kept_top(const uint path_index, const PathRecordPath p) {
    if (params.overlay_keep_update == 0) {
        return;
    }
    const uint key = path_debug_top_key(p);
    const uint bin = path_debug_key_bin(key);
    const uint band = state[PATH_DEBUG_STATE_BAND];
    if (key == 0 || bin < band) {
        return;
    }
    uint at;
    if (bin > band) {
        InterlockedAdd(state[PATH_DEBUG_STATE_CANDIDATES_ABOVE], 1, at);
        if (at < PATH_DEBUG_KEPT_SLOTS) {
            state[PATH_DEBUG_STATE_CANDIDATES + at] = path_index;
        }
        return;
    }
    InterlockedAdd(state[PATH_DEBUG_STATE_CANDIDATES_IN_BAND], 1, at);
    if (at < PATH_DEBUG_BAND_CANDIDATES) {
        state[PATH_DEBUG_STATE_CANDIDATES + PATH_DEBUG_KEPT_SLOTS + at] = path_index;
    }
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
    if (p.vertex_count == 0 || any(p.pixel >= path_debug_dim()) || !path_debug_matched(i)) {
        return;
    }
    if (path_debug_keeps_top()) {
        collect_kept_top(i, p);
    } else {
        path_debug_recent_commit(p);
    }
}
