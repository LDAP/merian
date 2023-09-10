#include "common/sampling.glsl"
#include "common/frames.glsl"

#ifndef _BSDF_VOLUME_H_
#define _BSDF_VOLUME_H_

#define bsdf_volume_isotropic_sample(random) sample_sphere(random)

// solid angle
#define bsdf_volume_isotropic_pdf() (INV_PI / 4)

// solid angle
#define bsdf_volume_isotropic_eval() vec3(INV_PI / 4)

#endif
