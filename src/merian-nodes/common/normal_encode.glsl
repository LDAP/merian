#ifndef _NORMAL_ENCODE_H_
#define _NORMAL_ENCODE_H_

vec2 sign_no0(const vec2 v)
{
  return mix(vec2(1.0), vec2(-1.0), lessThan(v, vec2(0.0)));
}

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
vec3 geo_decode_normal(const uint enc)
{
  const vec2 projected = unpackSnorm2x16(enc); // -1..1
  vec3 vec = vec3(projected, 1.0-abs(projected.x)-abs(projected.y));
  if (vec.z < 0.0) vec.xy = vec2(1.0-abs(vec.yx)) * sign_no0(vec.xy);
  return normalize(vec);
}

// 32-bit normal encoding from Journal of Computer Graphics Techniques Vol. 3, No. 2, 2014
// A Survey of Efficient Representations for Independent Unit Vectors,
// almost like oct30
uint geo_encode_normal(const vec3 n)
{
  const float invL1Norm = 1.0 / (abs(n.x) + abs(n.y) + abs(n.z));
  vec2 enc; // find floating point values of octahedral map in [-1,1]:
  if (n.z < 0.0) enc = (1.0-abs(n.yx * invL1Norm)) * sign_no0(n.xy);
  else           enc = n.xy * invL1Norm;
  return packSnorm2x16(enc);
}

vec3 geo_decode_normal_float(const float enc) {
  return geo_decode_normal(floatBitsToUint(enc));
}

float geo_encode_normal_float(const vec3 n) {
  return uintBitsToFloat(geo_encode_normal(n));
}



#endif
