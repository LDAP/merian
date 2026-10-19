import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

void threshold_fireflies() {
    const uint wanted = clamp(
        uint(float(state[PATH_DEBUG_STATE_MATCHED]) * params.overlay_brightest_fraction), 1u,
        max(params.overlay_max_paths, 1u));
    uint band = 0;
    uint band_keep = 0xFFFFFFFFu;
    uint above = 0;
    for (int bin = PATH_DEBUG_HIST_BINS - 1; bin >= 1; bin--) {
        const uint count = state[PATH_DEBUG_STATE_HIST + bin];
        if (above + count >= wanted) {
            band = uint(bin);
            band_keep = path_debug_share_threshold(count > 0 ? float(wanted - above) / float(count)
                                                             : 1.f);
            break;
        }
        above += count;
    }
    state[PATH_DEBUG_STATE_BAND] = band;
    state[PATH_DEBUG_STATE_BAND_KEEP] = band_keep;
}

void threshold_kept_top() {
    uint kept_count[PATH_DEBUG_HIST_BINS];
    for (uint bin = 0; bin < PATH_DEBUG_HIST_BINS; bin++) {
        kept_count[bin] = 0;
    }
    for (uint slot = 0; slot < params.overlay_max_paths; slot++) {
        const uint key = path_debug_kept_key_now(slot);
        if (key != 0) {
            kept_count[path_debug_key_bin(key)]++;
        }
    }
    uint band = 0;
    uint above = 0;
    for (int bin = PATH_DEBUG_HIST_BINS - 1; bin >= 0; bin--) {
        above += state[PATH_DEBUG_STATE_HIST + bin] + kept_count[bin];
        if (above >= params.overlay_max_paths) {
            band = uint(bin);
            break;
        }
    }
    state[PATH_DEBUG_STATE_BAND] = band;
}

[numthreads(1, 1, 1)]
[shader("compute")]
void main() {
    if (params.map_batch_end != 0) {
        stats[PATH_DEBUG_STATS_BATCHES]++;
        stats[PATH_DEBUG_STATS_BATCH_SAMPLES] = stats[PATH_DEBUG_STATS_SAMPLES];
        stats[PATH_DEBUG_STATS_BATCH_DRAWS] = stats[PATH_DEBUG_STATS_DRAWS];
    }

    if (params.overlay_rank == PATH_DEBUG_RANK_FIREFLIES) {
        threshold_fireflies();
    } else if (path_debug_keeps_top() && params.overlay_keep_update != 0) {
        threshold_kept_top();
    }
}
