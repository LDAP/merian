#include "merian-shaders/sampling.glsl"
#include "merian-shaders/frames.glsl"

#ifndef _MERIAN_SHADERS_PHASE_ISOTROPIC_H_
#define _MERIAN_SHADERS_PHASE_ISOTROPIC_H_

#define phase_isotropic_sample(random) sample_sphere(random)

// solid angle
#define phase_isotropic_pdf() (INV_PI / 4)

// solid angle
#define phase_isotropic_eval() vec3(INV_PI / 4)

#endif
