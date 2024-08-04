#ifndef _INTERPOLATE_H_
#define _INTERPOLATE_H_

// like smoothstep but without Hermite smoothing
float linearstep(const float edge0, const float edge1, const float x) {
  return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}

#endif
