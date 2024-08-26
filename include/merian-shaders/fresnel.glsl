#ifndef _MERIAN_SHADERS_FRESNEL_H_
#define _MERIAN_SHADERS_FRESNEL_H_

#define fresnel_schlick(minuswi_dot_h, F0) ((F0) + (1. - (F0)) * pow(max(1. - (minuswi_dot_h), 0), 5.))

#endif // _MERIAN_SHADERS_FRESNEL_H_
