#include "common/colors_yuv.glsl"
#include "common/colors_srgb.glsl"

#ifndef _TONEMAP_H_
#define _TONEMAP_H_

vec3 tonemap_clamp(const vec3 rgb) {
    return clamp(rgb, vec3(0), vec3(1));
}

vec3 tonemap_reinhard(const vec3 rgb, const float white) {
    float l = yuv_luminance(rgb);
    float l_new = (l * (1.0f + (l / (white * white)))) / (1.0f + l);
    return rgb * l_new / l;
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

vec3 tonemap_aces_approx(const vec3 rgb) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;

    const vec3 col = (rgb*(a*rgb+b))/(rgb*(c*rgb+d)+e);

    return clamp(col, 0., 1.);
}

// Lottes 2016, "Advanced Techniques and Optimization of HDR Color Pipelines" (AMD)
vec3 tonemap_lottes(const vec3 rgb, const float contrast, const float shoulder, const float _hdrMax, const float _midIn, const float _midOut) {
  const vec3 a = vec3(contrast);
  const vec3 d = vec3(shoulder);
  const vec3 hdrMax = vec3(_hdrMax);
  const vec3 midIn = vec3(_midIn);
  const vec3 midOut = vec3(_midOut);

  const vec3 b =
    (-pow(midIn, a) + pow(hdrMax, a) * midOut) /
    ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
  const vec3 c =
    (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
    ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

  return pow(rgb, a) / (pow(rgb, a * d) * b + c);
}

#endif
