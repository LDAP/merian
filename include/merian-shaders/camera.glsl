#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

#ifndef _MERIAN_SHADERS_CAMERA_H_
#define _MERIAN_SHADERS_CAMERA_H_

// importance sample the blackman harris pixel filter with 1.5px radius support
// result from turingbot, no exact solution.
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
  return halton_sequence[frame & 7] - .5;
}

// expects up, forward to be normalized
vec3 get_camera_ray_dir(const vec2 pixel, const vec2 resolution,
                        const vec3 up, const vec3 forward, const float fov_tan_alpha_half) {
  const vec3 uv = vec3((2. * pixel + 1.) / resolution - 1., 1.);
  const mat3x3 m = mat3x3(
    cross(forward, up),                // right
    -up * resolution.y / resolution.x, // up (normalized for aspect)
    forward / fov_tan_alpha_half
  );
  return normalize(m * uv);
}

// expects up, forward to be normalized
vec2 get_camera_pixel(const vec3 ray_dir, const vec2 resolution,
                      const vec3 up, const vec3 forward, const float fov_tan_alpha_half) {
  const mat3x3 m_ortho = mat3x3(cross(forward, up), up, forward);
  vec3 scale_inv = vec3(1.0, -resolution.x / resolution.y, fov_tan_alpha_half);
  vec3 uv = scale_inv * (ray_dir * m_ortho);
  uv.rg /= uv.b;
  return ((uv.rg + 1.) * resolution - 1) / 2.;
}

// Computes Exposure Value (EV) which is defined at ISO 100 from typical camera settings.
float ev_100(const float aperature, const float shutter_time, const float iso) {
  return log2((aperature * aperature * 100) / (shutter_time * iso));
}

// Computes Exposure Value (EV) which is defined at ISO 100 from average luminance.
// K is the reflected-light meter calibration constant.
float ev_100_from_average(const float avg_luminance, const float iso, const float K) {
  return log2(avg_luminance * iso / K);
}

// The maximum luminance without clipped or bloomed camera output (Saturation Based Sensitivity).
// q is the lens and vignetting attenuation (typical 0.65).
float ev_100_to_max_luminance(const float ev100, const float iso, const float q) {
  return 78.0 / (iso * q) * pow(2, ev100);
}

// This is what is multiplied to the color / luminance values to exposure.
float max_luminance_to_exposure(const float max_luminance) {
  // The maximum luminance without clipped or bloomed camera output (Saturation Based Sensitivity).
  return 1. / max_luminance;
}

#endif
