#pragma once

import merian_shaders.utils.random;
import merian_shaders.utils.sampling;

namespace merian {

struct Camera {
    float3 position;
    float3 target;
    float3 up;

    // normalized base vector, right. Normalized for FOV
    float3 U;
    // normalized base vector, up. Normalized for FOV
    float3 V;
    // normalized base vector, forward. Length == focal length
    float3 W;

    float near;
    float far;
    float aspect_ratio; // width / height
    float aperture_radius; // 0 = pinhole

    // Sub-pixel jitter in render-pixel units, typical range [-0.5, 0.5].
    // X-axis right, Y-axis down (matches DLSS / FSR2 input convention).
    float2 jitter;

    // ----------------------------------

    float2 pixel_to_ndc(const uint2 pixel, const uint2 frame_dimensions) {
        const float2 p = (pixel + float2(0.5f, 0.5f) + jitter) / frame_dimensions;
        return float2(2, -2) * p + float2(-1, 1);
    }

    RayDesc generate_ray_pinhole(const uint2 pixel, const uint2 frame_dimensions) {
        RayDesc ray;
        const float2 ndc = pixel_to_ndc(pixel, frame_dimensions);

        ray.Origin = position;
        ray.Direction = normalize(ndc.x * U + ndc.y * V + W);

        // account for curvature
        const float inv_cos = 1.f / dot(normalize(W), ray.Direction);
        ray.TMin = near * inv_cos;
        ray.TMax = far * inv_cos;

        return ray;
    }

    // Project a world-space position to fractional pixel coordinates using pinhole model.
    // U, V, W must be orthogonal (standard pinhole camera).
    float2 project_pinhole(const float3 world_pos, const uint2 frame_dimensions) {
        float3 d = world_pos - position;
        // t such that t*d lies on the focal plane (where W lives)
        float t = dot(W, W) / dot(d, W);
        float3 plane_pos = t * d - W;
        float2 ndc = float2(dot(plane_pos, U) / dot(U, U),
                            dot(plane_pos, V) / dot(V, V));
        // Inverse of pixel_to_ndc: ndc = float2(2,-2) * p + float2(-1,1)
        // => p = (ndc - float2(-1,1)) / float2(2,-2)
        float2 p = (ndc - float2(-1, 1)) / float2(2, -2);
        return p * float2(frame_dimensions) - float2(0.5, 0.5);
    }

    RayDesc generate_ray_thinlens(const uint2 pixel, const uint2 frame_dimensions, inout RandomGenerator rng) {
        RayDesc ray;
        const float2 ndc = pixel_to_ndc(pixel, frame_dimensions);

        ray.Origin = position;
        ray.Direction = ndc.x * U + ndc.y * V + W;

        float2 aperture_sample = sample_disk(rng);
        float3 ray_target = ray.Origin + ray.Direction;
        ray.Origin += aperture_radius * (aperture_sample.x * normalize(U) +
                                         aperture_sample.y * normalize(V));
        ray.Direction = normalize(ray_target - ray.Origin);

        float inv_cos = 1.f / dot(normalize(W), ray.Direction);
        ray.TMin = near * inv_cos;
        ray.TMax = far * inv_cos;

        return ray;
    }
}

}
