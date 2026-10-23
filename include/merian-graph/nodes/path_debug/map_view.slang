import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.colors.colorspaces;
import merian_shaders.colors.colormaps;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"
#include "merian-graph/nodes/path_debug/map_display.slangh"

RWTexture2D<float4> out_panels;

[numthreads(PATH_DEBUG_WG_2D, PATH_DEBUG_WG_2D, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint2 cell = gid.xy;
    const uint view = gid.z;
    if (any(cell >= params.map_res) || (params.map_panel_mask & (1u << view)) == 0) {
        return;
    }
    const uint2 atlas_cell =
        uint2(view % PATH_DEBUG_PANEL_ATLAS_COLUMNS, view / PATH_DEBUG_PANEL_ATLAS_COLUMNS);
    out_panels[atlas_cell * params.map_res + cell] = float4(path_debug_map_color(view, cell), 1.f);
}
