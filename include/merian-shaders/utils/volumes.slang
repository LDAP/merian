#pragma once

namespace merian {

// Beer-Lambert law / Beer's law
float transmittance(float optical_depth) {
    return exp(-optical_depth);
}

// Beer-Lambert law / Beer's law
// Distance t
float transmittance(float t, float mu_t) {
    return transmittance(t * mu_t);
}

// Beer-Lambert law / Beer's law
// Distance t
float transmittance(float t, float mu_t, float max_t) {
    return transmittance(min(t, max_t), mu_t);
}

// For homogeneous medium
// Returns a distance t
float transmittance_sample(float mu_t, float random) {
    return -log(1. - random) / mu_t;
}

// For transmittance_sample(mu_t, random) and a distance t.
float transmittance_pdf(float t, float mu_t) {
    return mu_t * transmittance(t, mu_t);
}

// For homogenous medium with max distance

// Returns the maxiumum uniform random value for maximum distance max_t.
float transmittance_xi_max(float max_t, float mu_t) {
    return (1. - exp(-max_t * mu_t));
}

// Like transmittance_sample(..., max_t) but allows to insert maxiumum uniform random value for
// maximum distance max_t.
float transmittance_sample2(float mu_t, float random, float xi_max) {
    return transmittance_sample(mu_t, xi_max * random);
}

// Like transmittance_pdf(..., max_t) but allows to insert maxiumum uniform random value for maximum
// distance max_t.
float transmittance_pdf2(float t, float mu_t, float xi_max) {
    return transmittance_pdf(t, mu_t) / xi_max;
}

float transmittance_sample(float mu_t, float random, float max_t) {
    return transmittance_sample2(mu_t, random, transmittance_xi_max(max_t, mu_t));
}

float transmittance_pdf(float t, float mu_t, float max_t) {
    transmittance_pdf2(t, mu_t, transmittance_xi_max(max_t, mu_t));
}

}
