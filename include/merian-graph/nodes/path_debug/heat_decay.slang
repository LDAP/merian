import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint slot = gid.x;
    if (slot >= params.grid_slots || grid[slot] == 0 ||
        grid[slot] == PATH_DEBUG_GRID_TOMBSTONE) {
        return;
    }

    float w = asfloat(grid[params.grid_slots + slot]) * params.heat_decay;
    if (w < 1e-8f) {
        grid[slot] = PATH_DEBUG_GRID_TOMBSTONE;
        w = 0.f;
    }
    grid[params.grid_slots + slot] = asuint(w);
    if (w > 0.f) {
        InterlockedAdd(state[PATH_DEBUG_STATE_GRID_CELLS], 1);
    }
    InterlockedMax(state[PATH_DEBUG_STATE_HEAT_MAX], asuint(w));
}
