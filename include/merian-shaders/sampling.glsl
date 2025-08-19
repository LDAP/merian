#extension GL_GOOGLE_include_directive: require

#include "common.glsl"
#include "random.glsl"

#ifndef _MERIAN_SHADERS_SAMPLING_H_
#define _MERIAN_SHADERS_SAMPLING_H_

// sample sphere, p = 1 / 4pi
vec3 sample_sphere(const vec2 random) {
   const float z = 2.0 * (random.x - 0.5);
   const float z2 = sqrt(1.0 - z * z);
   return vec3(z2 * cos(TWO_PI * random.y), z2 * sin(TWO_PI * random.y), z);
}

// sample hemisphere (around 0,0,1), p = 1 / 2pi
vec3 sample_hemisphere(const vec2 random) {
   const float su = sqrt(1 - (1 - random.x) * (1 - random.x));
   return vec3(su * cos(TWO_PI * random.y), su * sin(TWO_PI * random.y), 1 - random.x);
}

// sample hemisphere (around 0,0,1), cos lobe, p = cos(theta) / pi
vec3 sample_cos(const vec2 random) {
   const float su = sqrt(random.x);
   return vec3(su * cos(TWO_PI * random.y), su * sin(TWO_PI * random.y), sqrt(1.0 - random.x));
}

// samples barycentric coordinates of a triangle, p = 1 / A
vec3 sample_triangle(const vec2 random) {
   const float x = sqrt(random.x);

   return vec3(
      1 - x,
      x * (1 - random.y),
      x * random.y);
}

// samples coordinates on a unit disk, p = 1 / A
vec2 sample_disk(const vec2 random) {
   const float angle = TWO_PI * random.x;
   return vec2(cos(angle), sin(angle)) * sqrt(random.y);
}

// returns two pseudorandom normal distributed values.
vec2 sample_normal_box_muller(const float mean, const float sigma, const vec2 random) {
   // use 1.0 - random.x to ensure > eps
   return sigma * sqrt(-2.0 * log(1.0 - random.x)) * vec2(cos(TWO_PI * random.y), sin(TWO_PI * random.y)) + mean;
}

// init spare with NaN
float sample_normal_marsaglia(const float mean, const float sigma, inout uint rng_state, inout float spare) {
   if (!isnan(spare)) {
      const float res = spare * sigma + mean;
      spare = 0.0 / 0.0;
      return res;
   } else {
      vec2 u;
      float s;
      int i = 0;
      do {
         u = XorShift32Vec2(rng_state) * 2.0 - 1.0;
         s = u.x * u.x + u.y * u.y;
      } while ((s >= 1.0 || s == 0.0) && i++ < 10);
      s = sqrt(-2.0 * log(s) / s);
      spare = u.y * s;
      return u.x * s * sigma + mean;
   }
}

float sample_normal_pdf(const float mean, const float sigma, const float x) {
   const float z = (x - mean) / sigma;
   return INV_SQRT_TWO_PI * exp(-0.5 * z * z) / sigma;
}

#endif
