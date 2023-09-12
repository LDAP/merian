#include "common/sampling.glsl"
#include "common/frames.glsl"

#ifndef _PHASE_ISOTROPIC_H_
#define _PHASE_ISOTROPIC_H_

#define phase_isotropic_sample(random) sample_sphere(random)

// solid angle
#define phase_isotropic_pdf() (INV_PI / 4)

// solid angle
#define phase_isotropic_eval() vec3(INV_PI / 4)

#endif
