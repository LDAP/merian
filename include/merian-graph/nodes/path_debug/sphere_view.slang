import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.utils.encoding;
import merian_shaders.utils.frame;
import merian_shaders.colors.colorspaces;
import merian_shaders.colors.colormaps;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/map_display.slangh"

ParameterBlock<GBuffer> gbuffer;

// Draws the directional map onto a sphere floating at the selected pixel's primary hit, into the
// line overlay (under the path markers).
[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    if (params.sphere_view == 0) {
        return;
    }

    const float linear_z = gbuffer.get_linear_z(path_debug_selected_pixel());
    if (!(linear_z > 0.f) || linear_z > 1e6f) {
        return;
    }
    const float3 center =
        camera.position + camera.get_pinhole_direction(path_debug_selected_pixel(), path_debug_dim()) * linear_z;
    const float radius = params.sphere_radius * linear_z;

    // screen-space bounding square around the projected center; ndc.y spans 2*|V|/|W| at unit
    // depth along W
    const float2 c = camera.project_pinhole(center, path_debug_dim());
    const float pixel_radius = params.sphere_radius * length(camera.W) / length(camera.V) * 0.5f *
                                   float(params.dim_y) +
                               2.f;
    const int2 lo = int2(c - pixel_radius);
    const int2 p = lo + int2(gid.xy);
    if (any(p < 0) || any(p >= int2(path_debug_dim())) || any(gid.xy > uint(2.f * pixel_radius + 2.f))) {
        return;
    }

    // ray-sphere intersection
    const float3 rd = camera.get_pinhole_direction(uint2(p), path_debug_dim());
    const float3 oc = camera.position - center;
    const float b = dot(oc, rd);
    const float h2 = b * b - (dot(oc, oc) - radius * radius);
    if (h2 < 0.f) {
        return;
    }
    const float t = -b - sqrt(h2);
    if (t < 0.f) {
        return;
    }
    const float3 dir = normalize(camera.position + rd * t - center);

    float3x3 to_map = dir_map_frame(float3(0, 0, 1), false);
    if (merian_path_debug_gbuffer && params.map_frame == 1) {
        to_map = dir_map_frame(gbuffer.get_normal(path_debug_selected_pixel()), true);
    }
    float3 color = path_debug_view_display_dir(params.sphere_view - 1, mul(to_map, dir));

    // rim darkening as a cheap depth cue
    const float rim = saturate(-dot(dir, rd));
    color *= 0.35f + 0.65f * rim;

    InterlockedMax(overlay[p.y * int(params.dim_x) + p.x], path_debug_pack_overlay(200, color));
}
