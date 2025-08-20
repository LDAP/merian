#pragma once

#include "merian-shaders/utils/math-constants.slangh"

import "merian-shaders/utils/math.slang";

namespace merian {

// importance sample the blackman harris pixel filter with 1.5px radius support
// result from turingbot, no exact solution.
float2 pixel_offset_blackman_harris(float2 rand) {
    const float2 res = float2(cos(rand.y * MERIAN_PI * 2.0), sin(rand.y * MERIAN_PI * 2.0));
    // surprisingly good fit to inverse cdf
    const float r = 0.943404 * asin(0.636617 * asin(sqrt(rand.x)));
    return res * r;
}

float2 pixel_offset_halton(uint frame) {
    static const float2 halton_sequence[8] = {
        float2(1.0 / 2.0, 1.0 / 3.0), float2(1.0 / 4.0, 2.0 / 3.0), float2(3.0 / 4.0, 1.0 / 9.0),
        float2(1.0 / 8.0, 4.0 / 9.0), float2(5.0 / 8.0, 7.0 / 9.0), float2(3.0 / 8.0, 2.0 / 9.0),
        float2(7.0 / 8.0, 5.0 / 9.0), float2(1.0 / 16.0, 8.0 / 9.0)
    };
    return halton_sequence[frame & 7u] - 0.5f;
}

float3 get_camera_ray_dir(
    float2 pixel, float2 resolution, float3 up, float3 forward, float fov_tan_alpha_half) {

    float3 uv = float3((((pixel * 2.0f) + 1.0f) / resolution) - 1.0f, 1.0f);
    float3x3 m = float3x3(cross(forward, up),                    // right
                          ((-up) * resolution.y) / resolution.x, // up (normalized for aspect)
                          forward / fov_tan_alpha_half);
    return normalize(mul(uv, m));
}

float2 get_camera_pixel(
    float3 ray_dir, float2 resolution, float3 up, float3 forward, float fov_tan_alpha_half) {
    float3x3 m = float3x3(cross(forward, up),                    // right
                          ((-up) * resolution.y) / resolution.x, // up (normalized for aspect)
                          forward / fov_tan_alpha_half);
    float3 uv = mul(ray_dir, inverse(m));
    uv.rg /= uv.b;
    return ((uv.xy + 1.0f) * resolution - 1.0f.xx) / 2.0f;
}

float ev_100(float aperature, float shutter_time, float iso) {
    return log2(((aperature * aperature) * 100.0f) / (shutter_time * iso));
}

float ev_100_from_average(float avg_luminance, float iso, float K) {
    return log2((avg_luminance * iso) / K);
}

float ev_100_to_max_luminance(float ev100, float iso, float q) {
    return (78.0f / (iso * q)) * pow(2.0f, ev100);
}

float max_luminance_to_exposure(float max_luminance) {
    return 1.0f / max_luminance;
}

}
