import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

[numthreads(1, 1, 1)]
[shader("compute")]
void main() {
    if (params.map_batch_end != 0) {
        stats[PATH_DEBUG_STATS_BATCHES]++;
        stats[PATH_DEBUG_STATS_BATCH_SAMPLES] = stats[PATH_DEBUG_STATS_SAMPLES];
        stats[PATH_DEBUG_STATS_BATCH_DRAWS] = stats[PATH_DEBUG_STATS_DRAWS];
    }

    if (params.overlay_rank == PATH_DEBUG_RANK_ANY) {
        state[PATH_DEBUG_STATE_BAND_LO] = 0;
        state[PATH_DEBUG_STATE_BAND_HI] = PATH_DEBUG_HIST_BINS - 1;
        state[PATH_DEBUG_STATE_BAND_KEEP] = params.overlay_sample_max;
        return;
    }
    uint wanted = params.overlay_max_paths;
    if (params.overlay_rank == PATH_DEBUG_RANK_FIREFLIES) {
        wanted = clamp(uint(float(state[PATH_DEBUG_STATE_MATCHED]) *
                            params.overlay_brightest_fraction),
                       1u, max(params.overlay_max_paths, 1u));
    }
    uint band_lo = 0;
    uint band_hi = 0;
    uint band_keep = 0xFFFFFFFFu;
    uint above = 0;
    for (int bin = PATH_DEBUG_HIST_BINS - 1; bin >= 1; bin--) {
        const uint count = state[PATH_DEBUG_STATE_HIST + bin];
        if (above + count >= wanted) {
            band_lo = uint(bin);
            band_hi = uint(bin);
            const float keep = count > 0 ? float(wanted - above) / float(count) : 1.f;
            // 4294967040 is the largest float below 2^32
            band_keep = keep >= 1.f ? 0xFFFFFFFFu : uint(4294967040.f * keep);
            break;
        }
        above += count;
    }
    state[PATH_DEBUG_STATE_BAND_LO] = band_lo;
    state[PATH_DEBUG_STATE_BAND_HI] = band_hi;
    state[PATH_DEBUG_STATE_BAND_KEEP] = band_keep;
}
