#version 460

#extension GL_GOOGLE_include_directive : require

#ifndef _MERIAN_SHADERS_COLORS_XYZ_H_
#define _MERIAN_SHADERS_COLORS_XYZ_H_

float XYZ_luminance(const vec3 rgb) {
    return dot(rgb, vec3(0.2126729, 0.7151522, 0.0721750));
}

vec3 rgb_to_XYZ(const vec3 rgb) {
  const mat3 m_rgb_to_XYZ = (mat3(
    0.4124564, 0.2126729, 0.0193339,
    0.3575761, 0.7151522, 0.1191920,
    0.1804375, 0.0721750, 0.9503041
  ));
  return m_rgb_to_XYZ * rgb;
}

vec3 XYZ_to_rgb(const vec3 XYZ) {
  const mat3 m_XYZ_to_rgb = (mat3(
     3.2404542,-0.9692660, 0.0556434,
    -1.5371385, 1.8760108,-0.2040259,
    -0.4985314, 0.0415560, 1.0572252
  ));
  return m_XYZ_to_rgb * XYZ;
}

vec3 XYZ_to_xyY(const vec3 XYZ) {
    const float Y = XYZ.y;
    const float x = XYZ.x / (XYZ.x + XYZ.y + XYZ.z);
    const float y = XYZ.y / (XYZ.x + XYZ.y + XYZ.z);
    return vec3(x, y, Y);
}

// Converts a color from xyY space to XYZ space
vec3 xyY_to_XYZ(const vec3 xyY) {
    const float Y = xyY.z;
    const float x = Y * xyY.x / xyY.y;
    const float z = Y * (1.0 - xyY.x - xyY.y) / xyY.y;
    return vec3(x, Y, z);
}

#endif
