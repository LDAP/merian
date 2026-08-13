import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

// Applies the temporal decay, recycles dead cells, and collects the display maximum.
[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint slot = gid.x;
    if (slot >= params.grid_slots || grid[slot] == 0) {
        return;
    }

    float w = asfloat(grid[params.grid_slots + slot]) * params.heat_decay;
    if (w < 1e-8f) {
        grid[slot] = 0;
        w = 0.f;
    }
    grid[params.grid_slots + slot] = asuint(w);
    InterlockedMax(state[PATH_DEBUG_STATE_HEAT_MAX], asuint(w));
}
