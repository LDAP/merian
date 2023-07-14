#include "common/common.glsl"

#ifndef _CAMERA_H_
#define _CAMERA_H_

// importance sample the blackman harris pixel filter with 1.5px radius support
vec2 pixel_offset_blackman_harris(const vec2 rand) {
  const vec2 res = vec2(cos(rand.y * M_PI * 2.0),
                        sin(rand.y * M_PI * 2.0));
  // surprisingly good fit to inverse cdf
  const float r = 0.943404 * asin(0.636617 * asin(sqrt(rand.x)));
  return res * r;
}

vec2 pixel_offset_halton(const uint frame) {
  const vec2 halton_sequence[8] = {
    vec2(1.0 /  2.0, 1.0 / 3.0),
    vec2(1.0 /  4.0, 2.0 / 3.0),
    vec2(3.0 /  4.0, 1.0 / 9.0),
    vec2(1.0 /  8.0, 4.0 / 9.0),
    vec2(5.0 /  8.0, 7.0 / 9.0),
    vec2(3.0 /  8.0, 2.0 / 9.0),
    vec2(7.0 /  8.0, 5.0 / 9.0),
    vec2(1.0 / 16.0, 8.0 / 9.0)
  };
  return halton_sequence[frame & 7];
}

// expects up, forward to be normalized
vec3 get_camera_ray_dir(const vec2 pixel, const ivec2 resolution,
                        const vec3 up, const vec3 forward, const float fov_rad) {
  const mat2x3 m = mat2x3(
    cross(forward, up),                              // right
    -up * float(resolution.y) / float(resolution.x)  // up (normalized for aspect)
  );
  const vec2 uv = 2 * pixel / resolution - 1;
  return normalize(forward / tan(fov_rad / 2) + m * uv);
}

// expects up, forward to be normalized
vec2 get_camera_pixel(const vec3 ray_dir, const ivec2 resolution,
                      const vec3 up, const vec3 forward, const float fov_rad) {
  const mat3x3 m = mat3x3(
    cross(forward, up),                               // right
    -up * float(resolution.y) / float(resolution.x),  // up (normalized for aspect)
    forward / tan(fov_rad / 2)
  );
  vec3 uv = inverse(m) * ray_dir;
  return (uv.rg / uv.b + 1) * vec2(resolution) / 2;

  const float norm = tan(fov_rad / 2) * dot(ray_dir, forward);
  const vec3 norm_up = -up * float(resolution.y) / float(resolution.x);
  const vec3 right = cross(forward, up);
  const float u = dot(right, ray_dir) / norm;
  const float v = dot(norm_up, ray_dir) / norm;
  return (vec2(u, v) + 1) * vec2(resolution) / 2;
}

// // Returns a camera ray direction for a pixel. Expects up and forward to be normalized.
// // The direction is randomized using a blackman harris pixel filter.
// vec3 get_camera_ray_dir_bh(const ivec2 pixel, const ivec2 resolution, const vec3 up, const vec3 forward, const vec2 rand) {
//     const vec3 t = -up * float(resolution.y) / float(resolution.x);
//     const vec3 r = -normalize(cross(forward, t));
//     const vec2 off = filter_bh_sample(rand);
//     const vec2 uv = (pixel + off) / resolution - 0.5;
//     return normalize(0.45 * forward + r * uv.x + t * uv.y);
// }

// // Returns a camera ray direction for a pixel. Expects up and forward to be normalized.
// // The direction is offset by a halton sequence of lenght 8.
// vec3 get_camera_ray_dir_halton(const ivec2 pixel, const ivec2 resolution, const vec3 up, const vec3 forward, const uint frame) {
//     const vec2 halton_sequence[8] = {
//         vec2(1.0 /  2.0, 1.0 / 3.0),
//         vec2(1.0 /  4.0, 2.0 / 3.0),
//         vec2(3.0 /  4.0, 1.0 / 9.0),
//         vec2(1.0 /  8.0, 4.0 / 9.0),
//         vec2(5.0 /  8.0, 7.0 / 9.0),
//         vec2(3.0 /  8.0, 2.0 / 9.0),
//         vec2(7.0 /  8.0, 5.0 / 9.0),
//         vec2(1.0 / 16.0, 8.0 / 9.0)
//     };
//     const vec3 t = -up * float(resolution.y) / float(resolution.x);
//     const vec3 r = -normalize(cross(forward, t));
//     const vec2 uv = (pixel + halton_sequence[frame & 7]) / resolution - 0.5;
//     return normalize(0.45 * forward + r * uv.x + t * uv.y);
// }

#endif
