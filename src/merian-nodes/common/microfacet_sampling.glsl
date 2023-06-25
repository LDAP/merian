#include "common/common.glsl"

// sample sphere, p = 1/4pi
vec3 sample_sphere(vec2 x)
{
  float z = 2.0*(x.x-0.5);
  float z2 = sqrt(1.0-z*z);
  return vec3(z2 * cos(2.0*M_PI*x.y), z2 * sin(2.0*M_PI*x.y), z);
}

// sample hemisphere, cos lobe, p = cos(theta)/pi
vec3 sample_cos(vec2 x)
{
  float su = sqrt(x.x);
  return vec3(su*cos(2.0*3.1415*x.y), su*sin(2.0*3.1415*x.y), sqrt(1.0 - x.x));
}

vec3 bsdf_diffuse_sample(vec3 wi, vec3 du, vec3 dv, vec3 n, vec2 xi)
{
  return mat3(du, dv, n) * sample_cos(xi);
}

float bsdf_diffuse_pdf(vec3 wi, vec3 n, vec3 wo)
{
  return 1.0/M_PI;
}

float bsdf_diffuse_eval()
{
  return 1.0/M_PI;
}


//==================================================
// microfacet model by jonathan:
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
// L, V, N, Tx, Ty in world space
float bsdf_rough_eval(
    vec3 V, vec3 Tx, vec3 Ty, vec3 N, vec3 L, vec2 sigmaSq)
{
  V = -V; // all pointing away from surface intersection point
  vec3 H = normalize(L + V);
  float zetax = dot(H, Tx) / dot(H, N);
  float zetay = dot(H, Ty) / dot(H, N);

  float zL = dot(L, N); // cos of source zenith angle
  float zV = dot(V, N); // cos of receiver zenith angle
  float zH = dot(H, N); // cos of facet normal zenith angle
  if(zL < 0 || zV < 0 || zH < 0) return 0.0;
  float zH2 = zH * zH;

  float p = exp(-0.5 * (zetax * zetax / sigmaSq.x + zetay * zetay / sigmaSq.y))
    / (2.0 * M_PI * sqrt(sigmaSq.x * sigmaSq.y));

  float tanV = atan(dot(V, Ty), dot(V, Tx));
  float cosV2 = 1.0 / (1.0 + tanV * tanV);
  float sigmaV2 = sigmaSq.x * cosV2 + sigmaSq.y * (1.0 - cosV2);

  float tanL = atan(dot(L, Ty), dot(L, Tx));
  float cosL2 = 1.0 / (1.0 + tanL * tanL);
  float sigmaL2 = sigmaSq.x * cosL2 + sigmaSq.y * (1.0 - cosL2);

  float fresnel = 0.02 + 0.98 * pow(1.0 - dot(V, H), 5.0);

  zL = max(zL, 0.01);
  zV = max(zV, 0.01);

  return mix(1.0/M_PI, p / ((1.0 + Lambda(zL, sigmaL2) + Lambda(zV, sigmaV2)) * zV * zH2 * zH2 * 4.0), fresnel);
}
//==================================================

vec3 bsdf_sample(uint m, vec3 wi, vec3 du, vec3 dv, vec3 n, vec3 param, vec2 xi)
{
  if(m == 2) return sample_sphere(xi);
  return bsdf_diffuse_sample(wi, du, dv, n, xi);
}
float bsdf_pdf(uint m, vec3 wi, vec3 du, vec3 dv, vec3 n, vec3 wo, vec3 param)
{
  if(m == 2) return 1.0/(4.0*M_PI);
  return bsdf_diffuse_pdf(wi, n, wo);
}
float bsdf_eval(uint m, vec3 wi, vec3 du, vec3 dv, vec3 n, vec3 wo, vec3 param)
{ // evaluate *without* albedo, that has to multiplied in the end
  if(m == 2) return 1.0/(4.0*M_PI);
  if(m == 1) return bsdf_rough_eval(wi, du, dv, n, wo, param.xy);
  return bsdf_diffuse_eval();
}
