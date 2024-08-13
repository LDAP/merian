#ifndef _MERIAN_SHADERS_TRANSMITTANCE_H_
#define _MERIAN_SHADERS_TRANSMITTANCE_H_

// Beer-Lambert law / Beer's law

#define transmittance(optical_depth) (exp(-optical_depth))

#define transmittance2(distance, mu_t) transmittance(distance * mu_t)

// For homogeneous medium
#define transmittance_sample(mu_t, random) (-log(1. - random) / mu_t)

#define transmittance_pdf(distance, mu_t) (mu_t * transmittance2(distance, mu_t))

// For homogenous medium with max distance
#define transmittance3(distance, mu_t, max_distance) transmittance2(min(distance, max_distance), mu_t)

#define transmittance_xi_max(max_distance, mu_t) (1. - exp(-max_distance * mu_t))

#define transmittance_sample2(mu_t, random, xi_max) transmittance_sample(mu_t, xi_max * random)

#define transmittance_pdf2(distance, mu_t, xi_max) (transmittance_pdf(distance, mu_t) / xi_max)

#endif
