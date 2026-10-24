import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

// the camera the voxel depths were captured with
uniform merian::Camera voxel_camera;

// reversed z, a point at view depth w gets VOXEL_NEAR / w
static const float VOXEL_NEAR = 1e-4f;
// in pixel footprints
static const float MIN_DEPTH_SPAN = 1.f;
static const float MAX_DEPTH_SPAN = 32.f;
static const float4 CULLED = float4(2.f, 2.f, 2.f, 1.f);

struct Varyings {
    float4 position : SV_Position;
    nointerpolation uint source;
};

// unjittered, y down
float4 to_clip(const float3 pos) {
    const float3 d = pos - camera.position;
    return float4(dot(d, camera.U) / dot(camera.U, camera.U),
                  -dot(d, camera.V) / dot(camera.V, camera.V), VOXEL_NEAR,
                  dot(d, camera.W) / dot(camera.W, camera.W));
}

// The slice of a record pixel's frustum around its first hit, deep enough to cover the surface
// the pixel sees. A 14 vertex strip draws a box.
[shader("vertex")]
Varyings vertex_main(const uint vertex: SV_VertexID, const uint instance: SV_InstanceID) {
    Varyings out;
    out.position = CULLED;
    out.source = instance + 1;

    const uint2 pixel = uint2(instance % params.dim_x, instance / params.dim_x);
    const bool recorded =
        filtered[path_debug_filtered_base(pixel, false) + PATH_DEBUG_FILTERED_COUNT] != 0 ||
        moments[path_debug_moments_base(pixel) + PATH_DEBUG_MOMENT_COUNT] != 0;
    const float2 depth = voxel_depth[instance];
    if (!recorded || !(depth.x > 0.f)) {
        return out;
    }

    const uint2 dim = path_debug_dim();
    const float2 center = float2(pixel);
    const float footprint =
        depth.x *
        distance(voxel_camera.get_pinhole_direction_unjittered(center, dim),
                 voxel_camera.get_pinhole_direction_unjittered(center + float2(1.f, 0.f), dim));
    const float span = isfinite(depth.y) ? depth.y : 0.f;
    const float half_depth =
        0.5f * clamp(span, MIN_DEPTH_SPAN * footprint, MAX_DEPTH_SPAN * footprint);

    const uint bit = 1u << vertex;
    const float2 corner =
        float2((0x287au & bit) != 0 ? 0.5f : -0.5f, (0x02afu & bit) != 0 ? 0.5f : -0.5f);
    const float distance_along = depth.x + ((0x31e3u & bit) != 0 ? half_depth : -half_depth);
    const float3 dir = voxel_camera.get_pinhole_direction_unjittered(center + corner, dim);
    out.position = to_clip(voxel_camera.position + dir * max(distance_along, 0.f));
    return out;
}

[shader("fragment")]
uint fragment_main(const Varyings in) : SV_Target {
    return in.source;
}
