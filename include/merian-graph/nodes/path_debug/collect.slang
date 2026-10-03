import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint count = path_record_path_count(in_records);
    const uint i = gid.x;
    if (i >= count) {
        return;
    }

    const PathRecordPath p = path_record_load_path(in_records, i);
    if (p.vertex_count == 0) {
        return;
    }
    if (!path_debug_in_source(i, params.overlay_source)) {
        return;
    }
    if (params.overlay_rank == PATH_DEBUG_RANK_RESAMPLED) {
        path_debug_resample_commit(i, p);
        return;
    }
    if (params.overlay_rank == PATH_DEBUG_RANK_RECENT) {
        path_debug_recent_commit(p);
        return;
    }
    const uint bin = path_debug_lum_bin(p.luminance);
    if (bin < state[PATH_DEBUG_STATE_BAND_LO]) {
        return;
    }
    if (bin <= state[PATH_DEBUG_STATE_BAND_HI] &&
        pcg(path_record_pack_pixel(p.pixel)) > state[PATH_DEBUG_STATE_BAND_KEEP]) {
        return;
    }

    uint slot;
    InterlockedAdd(state[PATH_DEBUG_STATE_DRAW_COUNT], 1, slot);
    if (slot < params.overlay_max_paths && slot < PATH_DEBUG_MAX_DRAW) {
        state[PATH_DEBUG_STATE_DRAW_LIST + slot] = i;
    }
}
