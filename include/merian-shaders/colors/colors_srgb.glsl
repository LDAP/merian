

#extension GL_GOOGLE_include_directive : require

#ifndef _MERIAN_SHADERS_COLORS_SRGB_H_
#define _MERIAN_SHADERS_COLORS_SRGB_H_

// Converts the curve from sRGB to linear. Primary chromaticities are not transformed (output is close to rec709).
vec3 srgb_to_rgb(vec3 srgb) {
    const bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    const vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    const vec3 lower = srgb/vec3(12.92);

    return mix(higher, lower, cutoff);
}

// Converts the curve from linear to sRGB. Primary chromaticities are not transformed (input must be close to rec709).
vec3 rgb_to_srgb(vec3 rgb) {
    const bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
    const vec3 higher = vec3(1.055)*pow(rgb, vec3(1.0/2.4)) - vec3(0.055);
    const vec3 lower = rgb * vec3(12.92);

    return mix(higher, lower, cutoff);
}

#endif
