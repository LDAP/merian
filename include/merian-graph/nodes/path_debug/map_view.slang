import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.utils.encoding;
import merian_shaders.utils.frame;
import merian_shaders.colors.colorspaces;
import merian_shaders.colors.colormaps;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/map_display.slangh"

static const uint PANEL_MARGIN = 10;
static const uint PANEL_GAP = 26; // room for the overlay labels

// Renders the enabled directional-map panels into the output image in one dispatch:
// params.map_view is a bitmask of enabled views, gid.z selects the panel.
[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint2 p = gid.xy;
    if (any(p >= params.map_panel_size)) {
        return;
    }

    // gid.z-th set bit of the view mask
    uint view = 0;
    uint remaining = gid.z;
    for (; view < PATH_DEBUG_VIEW_COUNT; view++) {
        if ((params.map_view & (1u << view)) != 0) {
            if (remaining == 0) {
                break;
            }
            remaining--;
        }
    }
    if (view >= PATH_DEBUG_VIEW_COUNT) {
        return;
    }

    const float2 uv = (float2(p) + 0.5f) / float(params.map_panel_size);
    const float3 color = path_debug_view_display_uv(view, uv);

    const uint2 origin = uint2(params.dim_x - params.map_panel_size - PANEL_MARGIN,
                               PANEL_MARGIN + gid.z * (params.map_panel_size + PANEL_GAP));
    const uint2 target = origin + p;
    if (all(target < path_debug_dim())) {
        out_out[target] = float4(color, 1.f);
    }
}
