#pragma once

implementing colorspaces;

namespace merian {

float yuv_luminance(float3 rgb) {
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

half yuv_luminance(half3 rgb) {
    return dot(rgb, half3(float3(0.299, 0.587, 0.114)));
}

float3 rgb_to_yuv(float3 rgb) {
    return mul(rgb, float3x3(float3(0.299, -0.14713, 0.615), float3(0.587, -0.28886, -0.51499),
                             float3(0.114, 0.436, -0.10001)));
}

float3 yuv_to_rgb(float3 yuv) {
    return mul(yuv, float3x3(float3(1.0, 1.0, 1.0), float3(0.0, -0.39465, 2.03211),
                             float3(1.13983, -0.58060, 0.0)));
}

}
