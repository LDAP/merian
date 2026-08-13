import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

// Walks the luminance histogram from the top until the wanted number of paths is covered:
// k for top-k mode, a fraction of the selected paths for firefly mode.
[numthreads(1, 1, 1)]
[shader("compute")]
void main() {
    if (params.mode == PATH_DEBUG_MODE_SELECTED_PIXEL) {
        return;
    }
    uint wanted = params.k;
    if (params.mode == PATH_DEBUG_MODE_FIREFLIES) {
        wanted = max(1u, uint(float(state[PATH_DEBUG_STATE_SELECTED]) * params.firefly_fraction));
    }
    uint acc = 0;
    for (int bin = PATH_DEBUG_HIST_BINS - 1; bin >= 1; bin--) {
        acc += state[PATH_DEBUG_STATE_HIST + bin];
        if (acc >= wanted) {
            state[PATH_DEBUG_STATE_THRESHOLD] = asuint(path_debug_bin_lum_lower(uint(bin)));
            return;
        }
    }
    state[PATH_DEBUG_STATE_THRESHOLD] = asuint(0.f);
}
