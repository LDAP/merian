#pragma once

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

    float far;
    float near;
    float aspect_ratio; // width / height

    // ----------------------------------
}

}
