#include "merian-shaders/sampling.glsl"
#include "merian-shaders/frames.glsl"

#ifndef _MERIAN_SHADERS_BSDF_DIFFUSE_H_
#define _MERIAN_SHADERS_BSDF_DIFFUSE_H_

// Lambert diffuse BSDF

// n must be normalized
#define bsdf_diffuse_sample(n, random) (make_frame(n) * sample_cos(random))

// solid angle, not error checked against wdotn < 0.
#define bsdf_diffuse_pdf(wodotn) (INV_PI * wodotn)

// solid angle
#define bsdf_diffuse_eval(albedo) (INV_PI * albedo)

#endif
