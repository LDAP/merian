#ifndef _COLORS_SRGB_H_
#define _COLORS_SRGB_H_

vec3 srgb_to_rgb(vec3 srgb) {
    const bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    const vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    const vec3 lower = srgb/vec3(12.92);

    return mix(higher, lower, cutoff);
}

vec3 rgb_to_srgb(vec3 rgb) {
    const bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
    const vec3 higher = vec3(1.055)*pow(rgb, vec3(1.0/2.4)) - vec3(0.055);
    const vec3 lower = rgb * vec3(12.92);

    return mix(higher, lower, cutoff);
}

#endif
