#pragma once

implementing colorspaces;

// Converts the curve from sRGB to linear. Primary chromaticities are not transformed (output is
// close to rec709).
float3 srgb_to_rgb(float3 srgb) {
    const float3 higher = pow((srgb + 0.055) / 1.055, 2.4);
    const float3 lower = srgb / 12.92;

    // Component-wise conditional selection
    return select(srgb < 0.04045, lower, higher);
}

// Converts the curve from linear to sRGB. Primary chromaticities are not transformed (input must be
// close to rec709).
float3 rgb_to_srgb(float3 rgb) {
    const float3 higher = 1.055 * pow(rgb, 1.0 / 2.4) - 0.055;
    const float3 lower = rgb * 12.92;

    return select(rgb < 0.00313085, lower, higher);
}

