import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

// a cell below this weight is freed
static const float MIN_WEIGHT = 1e-8f;

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint slot = gid.x;
    if (slot >= params.grid_slots) {
        return;
    }
    const uint key = grid[slot];
    if (key == 0 || key == PATH_DEBUG_GRID_TOMBSTONE) {
        return;
    }

    const float decayed = asfloat(grid[params.grid_slots + slot]) * params.heat_decay;
    const bool faded = decayed < MIN_WEIGHT;
    if (faded) {
        grid[slot] = PATH_DEBUG_GRID_TOMBSTONE;
    }
    const float w = faded ? 0.f : decayed;
    grid[params.grid_slots + slot] = asuint(w);
    if (w > 0.f) {
        InterlockedAdd(state[PATH_DEBUG_STATE_GRID_CELLS], 1);
    }
    path_debug_atomic_max_float(state, PATH_DEBUG_STATE_HEAT_MAX, w);
}
