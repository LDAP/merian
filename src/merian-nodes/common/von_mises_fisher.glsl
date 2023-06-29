#include "common/common.glsl"

#ifndef _VON_MISES_FISHER_H_
#define _VON_MISES_FISHER_H_

// numerically robust von Mises Fisher lobe
float vmf_eval(float kappa, float dotmu)
{
  if(kappa < 1e-4) return 1.0/(4.0*M_PI);
  return kappa/(2.0*M_PI*(1.0 - exp(-2.0*kappa))) * exp(kappa*(dotmu-1.0));
}

vec3 vmf_sample(float kappa, vec2 r)
{ // see wenzel's doc on numerically stable expression for vmm:
  float w = 1.0 + log(r.x + (1.0-r.x)*exp(-2.0*kappa))/kappa;
  vec2 v = vec2(sin(2.0*M_PI*r.y), cos(2.0*M_PI*r.y));
  return vec3(sqrt(1.0-w*w)*v, w);
}

float vmf_get_kappa(float x)
{ // compute concentration parameter for given maximum density x
  if(x > 0.795) return 2.0*M_PI*x;
  return max(1e-5, (168.479*x*x + 16.4585*x - 2.39942)/
      (-1.12718*x*x+29.1433*x+1.0));
}

#endif
