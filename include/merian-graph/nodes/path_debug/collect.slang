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
        InterlockedAdd(state[PATH_DEBUG_STATE_CANDIDATES + 0], 1, at);
        if (at < PATH_DEBUG_KEPT_SLOTS) {
            state[PATH_DEBUG_STATE_DRAW_LIST + at] = path_index;
        }
        return;
    }
    InterlockedAdd(state[PATH_DEBUG_STATE_CANDIDATES + 1], 1, at);
    if (at < PATH_DEBUG_MAX_DRAW - PATH_DEBUG_KEPT_SLOTS) {
        state[PATH_DEBUG_STATE_DRAW_LIST + PATH_DEBUG_KEPT_SLOTS + at] = path_index;
    }
}

void collect_fireflies(const uint path_index, const PathRecordPath p) {
    const uint bin = path_debug_lum_bin(p.luminance);
    const uint band = state[PATH_DEBUG_STATE_BAND];
    if (bin < band ||
        (bin == band && path_debug_path_hash(p) > state[PATH_DEBUG_STATE_BAND_KEEP])) {
        return;
    }
    uint slot;
    InterlockedAdd(state[PATH_DEBUG_STATE_DRAW_COUNT], 1, slot);
    if (slot < params.overlay_max_paths && slot < PATH_DEBUG_MAX_DRAW) {
        state[PATH_DEBUG_STATE_DRAW_LIST + slot] = path_index;
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
    if (p.vertex_count == 0 || any(p.pixel >= path_debug_dim()) ||
        !path_debug_in_source(i, params.overlay_source)) {
        return;
    }
    switch (params.overlay_rank) {
    case PATH_DEBUG_RANK_FIREFLIES:
        collect_fireflies(i, p);
        break;
    case PATH_DEBUG_RANK_RECENT:
        path_debug_recent_commit(p);
        break;
    default:
        collect_kept_top(i, p);
        break;
    }
}
