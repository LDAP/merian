#include "common/common.glsl"

#ifndef _SAMPLING_H_
#define _SAMPLING_H_

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

#define sample_t_homogeneous_medium(mu_t, random) (-log(1. - random) / mu_t)

#define sample_t_homogeneous_medium_pdf(mu_t, t) (mu_t * exp(-mu_t * t))

#endif
