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

    // ----------------------------------

    float2 pixel_to_ndc(const uint2 pixel, const uint2 frame_dimensions) {
        const float2 p = (pixel + float2(0.5f, 0.5f)) / frame_dimensions;
        // if (applyJitter)
        //     p += float2(-data.jitterX, data.jitterY);
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
