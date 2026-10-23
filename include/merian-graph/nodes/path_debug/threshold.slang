import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

groupshared uint gs_kept[PATH_DEBUG_HIST_BINS];
groupshared uint gs_count[PATH_DEBUG_HIST_BINS];
groupshared uint gs_band_thread;

static const uint NO_BAND = 0xFFFFFFFFu;

// the band of the key histogram the kept top-k reaches into, over the kept and the new paths
[numthreads(PATH_DEBUG_HIST_BINS, 1, 1)]
[shader("compute")]
void main(uint3 tid: SV_GroupThreadID) {
    const uint t = tid.x;
    if (t == 0 && params.map_batch_end != 0) {
        stats[PATH_DEBUG_STATS_BATCHES]++;
        stats[PATH_DEBUG_STATS_BATCH_SAMPLES] = stats[PATH_DEBUG_STATS_SAMPLES];
        stats[PATH_DEBUG_STATS_BATCH_DRAWS] = stats[PATH_DEBUG_STATS_DRAWS];
    }
    if (!path_debug_keeps_top() || params.overlay_keep_update == 0) {
        return;
    }
    // brightest share starts over every frame
    const bool share = params.overlay_rank == PATH_DEBUG_RANK_BRIGHTEST_SHARE;

    gs_kept[t] = 0;
    if (t == 0) {
        gs_band_thread = NO_BAND;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint slot = t; !share && slot < params.overlay_max_paths; slot += PATH_DEBUG_HIST_BINS) {
        const uint key = path_debug_kept_key_now(slot);
        if (key != 0) {
            InterlockedAdd(gs_kept[path_debug_key_bin(key)], 1);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // brightest first
    const uint bin = PATH_DEBUG_HIST_BINS - 1 - t;
    const uint count = state[PATH_DEBUG_STATE_HIST + bin] + gs_kept[bin];
    gs_count[t] = count;
    GroupMemoryBarrierWithGroupSync();
    uint brighter = 0;
    for (uint u = 0; u < t; u++) {
        brighter += gs_count[u];
    }

    const uint wanted =
        share ? clamp(uint(float(state[PATH_DEBUG_STATE_MATCHED]) * params.overlay_brightest_share),
                      1u, params.overlay_max_paths)
              : params.overlay_max_paths;
    if (brighter + count >= wanted) {
        InterlockedMin(gs_band_thread, t);
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) {
        state[PATH_DEBUG_STATE_KEEP_COUNT] = wanted;
        if (gs_band_thread == NO_BAND) {
            state[PATH_DEBUG_STATE_BAND] = 0;
        }
    }
    if (t == gs_band_thread) {
        state[PATH_DEBUG_STATE_BAND] = bin;
    }
}
