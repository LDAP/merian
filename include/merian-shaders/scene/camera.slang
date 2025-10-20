#pragma once

import merian_shaders.utils.random;
import merian_shaders.utils.sampling;

namespace merian {

struct Camera {
    float3 position;
    float3 target;
    float3 up;
    float3 prev_position;

    // normalized base vector, right. Normalized for FOV
    float3 U;
    // normalized base vector, up. Normalized for FOV
    float3 V;
    // normalized base vector, forward. Length == focal length
    float3 W;

    float near;
    float far;
    float aspect_ratio; // width / height

    // ----------------------------------

    float3 generate_ray_dir_pinhole_non_normalized(const uint2 pixel,
                                                   const uint2 frame_dimensions) {
        const float2 p = (pixel + float2(0.5f, 0.5f)) / frame_dimensions;
        // if (applyJitter)
        //     p += float2(-data.jitterX, data.jitterY);
        const float2 ndc = float2(2, -2) * p + float2(-1, 1);

        return ndc.x * U + ndc.y * V + W;
    }

    RayDesc generate_ray_pinhole(const uint2 pixel, const uint2 frame_dimensions) {
        RayDesc ray;

        ray.Origin = position;
        ray.Direction = normalize(generate_ray_dir_pinhole_non_normalized(pixel, frame_dimensions));

        // account for curvature
        const float inv_cos = 1.f / dot(normalize(W), ray.Direction);
        ray.TMin = near * inv_cos;
        ray.TMax = far * inv_cos;

        return ray;
    }

    RayDesc compute_ray_thinlens(const uint2 pixel, const uint2 frame_dimensions, inout RandomGenerator rng) {
        RayDesc ray;

        const float2 p = (pixel + float2(0.5f, 0.5f)) / frame_dimensions;
        // if (applyJitter)
        //     p += float2(-data.jitterX, data.jitterY);
        const float2 ndc = float2(2, -2) * p + float2(-1, 1);

        ray.Origin = position;

        ray.Direction = ndc.x * U + ndc.y * V + W;
        float2 apertureSample = sample_disk(rng.next_float2()); // Sample lies in the unit disk [-1,1]^2
        float3 rayTarget = ray.origin + ray.dir;
        ray.origin += data.apertureRadius * (apertureSample.x * normalize(data.cameraU) +
                                             apertureSample.y * normalize(data.cameraV));
        ray.dir = normalize(rayTarget - ray.origin);

        float invCos = 1.f / dot(normalize(data.cameraW), ray.dir);
        ray.tMin = data.nearZ * invCos;
        ray.tMax = data.farZ * invCos;

        return ray;
    }
}

}
