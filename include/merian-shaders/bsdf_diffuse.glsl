#include "merian-shaders/sampling.glsl"
#include "merian-shaders/frames.glsl"

#ifndef _BSDF_DIFFUSE_H_
#define _BSDF_DIFFUSE_H_

// n must be normalized
#define bsdf_diffuse_sample(n, random) (make_frame(n) * sample_cos(random))

// solid angle
#define bsdf_diffuse_pdf(wodotn) (INV_PI * wodotn)

// solid angle
#define bsdf_diffuse_eval(albedo) (INV_PI * albedo)

#endif
