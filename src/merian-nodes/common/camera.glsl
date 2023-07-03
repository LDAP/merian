#include "common/common.glsl"

#ifndef _CAMERA_H_
#define _CAMERA_H_

// importance sample the blackman harris pixel filter.
// has 1.5px radius support
vec2 filter_bh_sample(const vec2 rand) {
  const vec2 res = vec2(cos(rand.y*M_PI*2.0), sin(rand.y*M_PI*2.0));
  const float r = 0.943404 * asin(0.636617 * asin(sqrt(rand.x))); // surprisingly good fit to inverse cdf
  return res * r;
}

// Returns a camera ray direction for a pixel. Expects up and forward to be normalized.
// The direction is randomized using a blackman harris pixel filter.
vec3 get_camera_ray_dir(const ivec2 pixel, const ivec2 resolution, const vec3 up, const vec3 forward, const vec4 rand) {
    const vec3 t = -up * float(resolution.y) / float(resolution.x);
    const vec3 r = -normalize(cross(forward, t));
    const vec2 off = filter_bh_sample(rand.yz);
    const vec2 uv = (pixel + off) / resolution - 0.5;
    return normalize(0.45 * forward + r * uv.x + t * uv.y);
}


#endif
