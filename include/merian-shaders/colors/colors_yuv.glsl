#version 460

#extension GL_GOOGLE_include_directive : require

#ifndef _MERIAN_SHADERS_COLORS_YUV_H_
#define _MERIAN_SHADERS_COLORS_YUV_H_

#define yuv_luminance(rgb) dot(rgb, vec3(0.299, 0.587, 0.114))

#define yuv_luminance_f16(rgb) dot(rgb, f16vec3(0.299hf, 0.587hf, 0.114hf))

vec3 rgb_to_yuv(const vec3 rgb) {
    const mat3 M = mat3(0.299, -0.14713,  0.615,
                        0.587, -0.28886, -0.51499,
                        0.114,  0.436,   -0.10001);
    return M * rgb;
}

vec3 yuv_to_rgb(vec3 yuv) {
    const mat3 M = mat3(1.0,      1.0,     1.0,
                        0.0,     -0.39465, 2.03211,
                        1.13983, -0.58060, 0.0);
    return M * yuv;
}

#endif
