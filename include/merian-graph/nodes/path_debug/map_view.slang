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

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint2 p = gid.xy;
    if (any(p >= params.map_panel_size)) {
        return;
    }

    uint view = 0;
    uint remaining = gid.z;
    for (; view < PATH_DEBUG_MAP_CHANNEL_COUNT; view++) {
        if ((params.map_panel_mask & (1u << view)) != 0) {
            if (remaining == 0) {
                break;
            }
            remaining--;
        }
    }
    if (view >= PATH_DEBUG_MAP_CHANNEL_COUNT) {
        return;
    }

    const int2 origin =
        int2(int(params.dim_x - params.map_panel_size - PATH_DEBUG_PANEL_MARGIN),
             int(PATH_DEBUG_PANEL_MARGIN +
                 gid.z * (params.map_panel_size + PATH_DEBUG_PANEL_GAP)) -
                 int(params.map_panel_scroll));
    const int2 target = origin + int2(p);
    if (any(target < 0) || any(target >= int2(path_debug_dim()))) {
        return;
    }

    const uint close_x = params.map_panel_size - PATH_DEBUG_PANEL_CLOSE;
    if (p.x >= close_x && p.y < PATH_DEBUG_PANEL_CLOSE) {
        const float2 c = float2(p) - float2(float(close_x), 0.f) -
                         (PATH_DEBUG_PANEL_CLOSE * 0.5f - 0.5f);
        const float arm = max(abs(abs(c.x) - abs(c.y)), abs(c.x) + abs(c.y) - 4.f);
        out_out[uint2(target)] = float4(arm < 1.f ? float3(1.f) : float3(0.15f), 1.f);
        return;
    }

    const float2 uv = (float2(p) + 0.5f) / float(params.map_panel_size);
    out_out[uint2(target)] = float4(path_debug_map_color_uv(view, uv), 1.f);
}
