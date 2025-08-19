#version 460

#extension GL_GOOGLE_include_directive : require

#include "merian-shaders/color/colors_yuv.glsl"
#include "merian-shaders/color/colors_srgb.glsl"
#include "merian-shaders/common.glsl"

#ifndef _MERIAN_SHADERS_TONEMAP_H_
#define _MERIAN_SHADERS_TONEMAP_H_

vec3 tonemap_clamp(const vec3 rgb) {
    return clamp(rgb, vec3(0), vec3(1));
}

vec3 tonemap_reinhard(const vec3 rgb, const float white) {
    float l = yuv_luminance(rgb);
    float l_new = (l * (1.0f + (l / (white * white)))) / (1.0f + l);
    return rgb * l_new * MERIAN_SAFE_RECIPROCAL(l);
}

vec3 _tonemap_uncharted2(const vec3 x) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;

    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

// Hable Tone Mapping / Hable Filmic
vec3 tonemap_uncharted2(const vec3 rgb, const float exposure_bias, const float W) {
    return _tonemap_uncharted2(rgb * exposure_bias) / _tonemap_uncharted2(vec3(W));
}

// Adapted from Stephen Hill (@self_shadow)
vec3 tonemap_aces(const vec3 rgb) {
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    const mat3 IN = mat3(
      0.59719, 0.07600, 0.02840,
      0.35458, 0.90834, 0.13383,
      0.04823, 0.01566, 0.83777);

    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    const mat3 OUT = mat3(
      1.60475, -0.10208, -0.00327,
      -0.53108,  1.10813, -0.07276,
      -0.07367, -0.00605,  1.07602);

    vec3 col = IN * rgb;

    const vec3 a = col * (col + 0.0245786f) - 0.000090537f;
    const vec3 b = col * (0.983729f * col + 0.4329510f) + 0.238081f;
    col = a / b;

    return clamp(OUT * col, 0., 1.);
}

/* 
Default parameters:
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
*/
vec3 tonemap_aces_approx(const vec3 rgb, const float a, const float b, const float c, const float d, const float e) {
    const vec3 col = (rgb*(a*rgb+b))/(rgb*(c*rgb+d)+e);
    return clamp(col, 0., 1.);
}

// Lottes 2016, "Advanced Techniques and Optimization of HDR Color Pipelines" (AMD)
// https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/d7531ae47d8b36a5d4025663e731a47a38be882f/framework/cauldron/framework/inc/shaders/tonemapping/tonemappers.hlsl#L21
vec3 tonemap_lottes(vec3 rgb, const float contrast, const float shoulder, const float hdrMax, const float midIn, const float midOut) {
    const float b = -((-pow(midIn, contrast) + (midOut*(pow(hdrMax, contrast*shoulder)*pow(midIn, contrast) -
            pow(hdrMax, contrast)*pow(midIn, contrast*shoulder)*midOut)) /
            (pow(hdrMax, contrast*shoulder)*midOut - pow(midIn, contrast*shoulder)*midOut)) /
            (pow(midIn, contrast*shoulder)*midOut));
    const float c = (pow(hdrMax, contrast*shoulder)*pow(midIn, contrast) - pow(hdrMax, contrast)*pow(midIn, contrast*shoulder)*midOut) /
           (pow(hdrMax, contrast*shoulder)*midOut - pow(midIn, contrast*shoulder)*midOut);

    rgb = min(rgb, vec3(hdrMax)); // fix for shoulder > 1
    float peak = max(rgb.r, max(rgb.g, rgb.b));

    peak = max(1e-7f, peak);

    vec3 ratio = rgb / peak;
    const float z = pow(peak, contrast);
    peak = z / (pow(z, shoulder) * b + c);

    const float crosstalk = 4.0; // controls amount of channel crosstalk
    const float saturation = contrast; // full tonal range saturation control
    const float crossSaturation = contrast * 16.0; 

    float white = 1.0;

    ratio = pow(abs(ratio), vec3(saturation / crossSaturation));
    ratio = mix(ratio, vec3(white), pow(peak, crosstalk));
    ratio = pow(abs(ratio), vec3(crossSaturation));

    return peak * ratio;
}

#endif
