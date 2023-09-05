#include "common/common.glsl"
#include "common/sampling.glsl"

/*
 * microfacet model by jonathan:
 */

#ifndef _BSDF_MICROFACET_H_
#define _BSDF_MICROFACET_H_

//-----------------------------------------------------------------

float erfc(float x) {
  return 2.0 * exp(-x * x) / (2.319 * x + sqrt(4.0 + 1.52 * x * x));
}

float erf(const float x) {
  const float a  = 0.140012;
  const float x2 = x*x;
  const float ax2 = a*x2;
  return sign(x) * sqrt( 1.0 - exp(-x2*(4.0/M_PI + ax2)/(1.0 + ax2)) );
}

float Lambda(const float cosTheta, const float sigmaSq) {
  const float v = cosTheta / sqrt((1.0 - cosTheta * cosTheta) * (2.0 * sigmaSq));
  return max(0.0, (exp(-v * v) - v * sqrt(M_PI) * erfc(v)) / (2.0 * v * sqrt(M_PI)));
  //return (exp(-v * v)) / (2.0 * v * sqrt(M_PI)); // approximate, faster formula
}

//-----------------------------------------------------------------

vec3 bsdf_microfacet_sample(const vec3 wi, const vec3 du, const vec3 dv, const vec3 n, const vec2 random) {
    // this is just regular diffuse sampling for now
    return mat3(du, dv, n) * sample_cos(random);
}

float bsdf_microfacet_pdf(const vec3 wi, const vec3 n, const vec3 wo) {
    // this is just regular diffuse pdf for now
    return INV_PI;
}

// -wi, du, dv, n, wo in world space
float bsdf_microfacet_eval(const vec3 minus_wi, const vec3 du, const vec3 dv, const vec3 n, const vec3 wo, const vec2 sigmaSq) {
  const vec3 H = normalize(wo + minus_wi);
  const float zetax = dot(H, du) / dot(H, n);
  const float zetay = dot(H, dv) / dot(H, n);

  float zL = dot(wo, n); // cos of source zenith angle
  float zV = dot(minus_wi, n); // cos of receiver zenith angle
  const float zH = dot(H, n); // cos of facet normal zenith angle
  if (zL < 0 || zV < 0 || zH < 0) return 0.0;
  const float zH2 = zH * zH;

  const float p = exp(-0.5 * (zetax * zetax / sigmaSq.x + zetay * zetay / sigmaSq.y))
    / (2.0 * M_PI * sqrt(sigmaSq.x * sigmaSq.y));

  const float tanV = atan(dot(minus_wi, dv), dot(minus_wi, du));
  const float cosV2 = 1.0 / (1.0 + tanV * tanV);
  const float sigmaV2 = sigmaSq.x * cosV2 + sigmaSq.y * (1.0 - cosV2);

  const float tanL = atan(dot(wo, dv), dot(wo, du));
  const float cosL2 = 1.0 / (1.0 + tanL * tanL);
  const float sigmaL2 = sigmaSq.x * cosL2 + sigmaSq.y * (1.0 - cosL2);

  // Schlick's approximation
  const float fresnel = 0.02 + 0.98 * pow(1.0 - dot(minus_wi, H), 5.0);

  zL = max(zL, 0.01);
  zV = max(zV, 0.01);

  return mix(INV_PI, p / ((1.0 + Lambda(zL, sigmaL2) + Lambda(zV, sigmaV2)) * zV * zH2 * zH2 * 4.0), fresnel);
}

#endif
