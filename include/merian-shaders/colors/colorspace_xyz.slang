#pragma once

implementing colorspaces;

namespace merian {

float XYZ_luminance(float3 rgb) {
    return dot(rgb, float3(0.2126729, 0.7151522, 0.0721750));
}

float3 rgb_to_XYZ(float3 rgb) {
    return mul(rgb, float3x3(float3(0.4124564, 0.2126729, 0.0193339),
                             float3(0.3575761, 0.7151522, 0.1191920),
                             float3(0.1804375, 0.0721750, 0.9503041)));
}

float3 XYZ_to_rgb(float3 XYZ) {
    return mul(XYZ, float3x3(float3(3.2404542,-0.9692660, 0.0556434),
                             float3(-1.5371385, 1.8760108,-0.2040259),
                             float3(-0.4985314, 0.0415560, 1.0572252)));
}

float3 XYZ_to_xyY(float3 XYZ) {
    float Y = XYZ.y;
    float x = XYZ.x / ((XYZ.x + XYZ.y) + XYZ.z);
    float y = XYZ.y / ((XYZ.x + XYZ.y) + XYZ.z);
    return float3(x, y, Y);
}

float3 xyY_to_XYZ(float3 xyY) {
    float Y = xyY.z;
    float x = (Y * xyY.x) / xyY.y;
    float z = (Y * ((1.0f - xyY.x) - xyY.y)) / xyY.y;
    return float3(x, Y, z);
}

}
