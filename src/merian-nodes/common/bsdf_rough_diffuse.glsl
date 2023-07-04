#include "common/common.glsl"
#include "common/sampling.glsl"

/*
 * microfacet model by jonathan:
 */

#ifndef _BSDF_ROUGH_DIFFUSE_H_
#define _BSDF_ROUGH_DIFFUSE_H_

//-----------------------------------------------------------------

float erfc(float x) {
  return 2.0 * exp(-x * x) / (2.319 * x + sqrt(4.0 + 1.52 * x * x));
}

float erf(float x) {
  float a  = 0.140012;
  float x2 = x*x;
  float ax2 = a*x2;
  return sign(x) * sqrt( 1.0 - exp(-x2*(4.0/M_PI + ax2)/(1.0 + ax2)) );
}

float Lambda(float cosTheta, float sigmaSq) {
  float v = cosTheta / sqrt((1.0 - cosTheta * cosTheta) * (2.0 * sigmaSq));
  return max(0.0, (exp(-v * v) - v * sqrt(M_PI) * erfc(v)) / (2.0 * v * sqrt(M_PI)));
  //return (exp(-v * v)) / (2.0 * v * sqrt(M_PI)); // approximate, faster formula
}

//-----------------------------------------------------------------

vec3 bsdf_rough_diffuse_sample(const vec3 wi, const vec3 du, const vec3 dv, const vec3 n, const vec2 random) {
    // this is just regular diffuse sampling
    return mat3(du, dv, n) * sample_cos(random);
}

float bsdf_rough_diffuse_pdf(const vec3 wi, const vec3 n, const vec3 wo) {
    // this is just regular diffuse pdf
    return 1.0 / M_PI;
}

// wi, du, dv, n, wo in world space
float bsdf_rough_diffuse_eval(vec3 wi, const vec3 du, const vec3 dv, const vec3 n, const vec3 wo, const vec2 sigmaSq) {
  wi = -wi; // all pointing away from surface intersection point
  vec3 H = normalize(wo + wi);
  float zetax = dot(H, du) / dot(H, n);
  float zetay = dot(H, dv) / dot(H, n);

  float zL = dot(wo, n); // cos of source zenith angle
  float zV = dot(wi, n); // cos of receiver zenith angle
  float zH = dot(H, n); // cos of facet normal zenith angle
  if(zL < 0 || zV < 0 || zH < 0) return 0.0;
  float zH2 = zH * zH;

  float p = exp(-0.5 * (zetax * zetax / sigmaSq.x + zetay * zetay / sigmaSq.y))
    / (2.0 * M_PI * sqrt(sigmaSq.x * sigmaSq.y));

  float tanV = atan(dot(wi, dv), dot(wi, du));
  float cosV2 = 1.0 / (1.0 + tanV * tanV);
  float sigmaV2 = sigmaSq.x * cosV2 + sigmaSq.y * (1.0 - cosV2);

  float tanL = atan(dot(wo, dv), dot(wo, du));
  float cosL2 = 1.0 / (1.0 + tanL * tanL);
  float sigmaL2 = sigmaSq.x * cosL2 + sigmaSq.y * (1.0 - cosL2);

  float fresnel = 0.02 + 0.98 * pow(1.0 - dot(wi, H), 5.0);

  zL = max(zL, 0.01);
  zV = max(zV, 0.01);

  return mix(1.0/M_PI, p / ((1.0 + Lambda(zL, sigmaL2) + Lambda(zV, sigmaV2)) * zV * zH2 * zH2 * 4.0), fresnel);
}

#endif
