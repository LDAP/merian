#pragma once

namespace merian {

// Returns t such that ray_origin + t * ray_dir = intersection
float intersect_plane(const float3 plane_origin,
                      const float3 plane_normal,
                      const float3 ray_origin,
                      const float3 ray_dir) {
    return dot(plane_normal, plane_origin - ray_origin) / dot(plane_normal, ray_dir);
}

}
