
#extension GL_GOOGLE_include_directive : require

#ifndef _MERIAN_SHADERS_COLORS_OKLCH_H_
#define _MERIAN_SHADERS_COLORS_OKLCH_H_

#include "colors_oklab.glsl"

// h in range 0, 2pi
vec3 oklch_to_rgb(const vec3 lch) {
    return oklab_to_rgb(vec3(lch.x, lch.y * cos(lch.z), lch.y * sin(lch.z)));
}

vec3 rgb_to_oklch(const vec3 rgb) {
    const vec3 oklab = rgb_to_oklab(rgb);
    return vec3(oklab.x, sqrt(MERIAN_SQUARE(oklab.y) * MERIAN_SQUARE(oklab.z)), atan(oklab.z, oklab.y));
}

#endif // _MERIAN_SHADERS_COLORS_OKLCH_H_
