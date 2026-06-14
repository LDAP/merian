#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.utils.math;

namespace merian {

/**
 *  F0: Specular reflectance at normal incidence (0 degrees); specular albedo;
 *  pow2((n1-n2)/(n1+n2)).
 *  F90: Specular reflectance at orthogonal incidence (90 degrees); 1.0 for specular reflection.
 *  cos_theta: widoth, cosine of angle between microfacet normal (h) and light
 *  direction (wi).
 */
T fresnel_schlick<T : IFloat>(const float cos_theta, const T F0, const T F90) {
    return (F0) + (F90 - F0) * T(pow(max(1. - (cos_theta), 0), 5.));
}

/**
 *  F0: Specular reflectance at normal incidence (0 degrees); specular albedo;
 *  pow2((n1-n2)/(n1+n2)).
 *  cos_theta: widoth, cosine of angle between microfacet normal (h) and light
 *  direction (wi).
 */
T fresnel_schlick<T : IFloat>(const float cos_theta, const T F0) {
    return fresnel_schlick(cos_theta, F0, T(1.));
}

/**
 * Unpolarized Fresnel reflectance at a dielectric interface. eta is the relative index of
 * refraction (incident / transmitted), cos_i is the cosine of the incidence angle (>= 0).
 * Returns 1.0 on total internal reflection.
 */
float fresnel_dielectric(const float cos_i, const float eta) {
    const float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
    if (sin2_t >= 1.0f) {
        return 1.0f;
    }
    const float cos_t = sqrt(1.0f - sin2_t);
    const float r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    const float r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5f * (r_parl * r_parl + r_perp * r_perp);
}

// --- Thin-film iridescence (Belcour & Barla 2017, "A Practical Extension to Microfacet Theory
// for the Modeling of Varying Iridescence") ---

float iridescence_ior_to_f0(const float transmitted_ior, const float incident_ior) {
    return pow2((transmitted_ior - incident_ior) / (transmitted_ior + incident_ior));
}

float3 iridescence_ior_to_f0(const float3 transmitted_ior, const float incident_ior) {
    const float3 r = (transmitted_ior - incident_ior) / (transmitted_ior + incident_ior);
    return r * r;
}

float3 iridescence_f0_to_ior(const float3 f0) {
    const float3 s = sqrt(f0);
    return (1.0f + s) / (1.0f - s);
}

// Spectral sensitivity of the eye to an optical path difference, evaluated in sRGB (Rec.709).
float3 iridescence_sensitivity(const float opd, const float3 shift) {
    const float phase = 2.0f * MERIAN_PI * opd * 1.0e-9f;
    const float3 val = float3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
    const float3 pos = float3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
    const float3 var = float3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);
    float3 xyz =
        val * sqrt(2.0f * MERIAN_PI * var) * cos(pos * phase + shift) * exp(-phase * phase * var);
    xyz.x += 9.7470e-14f * sqrt(2.0f * MERIAN_PI * 4.5282e+09f) *
             cos(2.2399e+06f * phase + shift.x) * exp(-4.5282e+09f * phase * phase);
    xyz /= 1.0685e-7f;
    return float3(3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
                  -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
                  0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
}

// RGB reflectance of a thin film of `thickness` (nm) and index `film_ior` on a base of reflectance
// `base_f0`, seen through a medium of index `outside_ior` at incidence cosine `cos_theta1`.
float3 fresnel_iridescent(const float outside_ior,
                          const float film_ior,
                          const float3 base_f0,
                          const float thickness,
                          const float cos_theta1) {
    // Fade the film index towards the surrounding medium as the film vanishes.
    const float iri_ior = lerp(outside_ior, film_ior, smoothstep(0.0f, 0.03f, thickness));

    // Refraction cosine into the film (Snell).
    const float sin_theta2_sq = pow2(outside_ior / iri_ior) * (1.0f - pow2(cos_theta1));
    const float cos_theta2_sq = 1.0f - sin_theta2_sq;
    // Full reflectance on total internal reflection; the negated test also rejects a non-finite
    // cosine (e.g. a degenerate half-vector) so the terms below stay finite.
    if (!(cos_theta2_sq > 0.0f)) {
        return float3(1.0f);
    }
    const float cos_theta2 = sqrt(cos_theta2_sq);

    // First interface (outside | film).
    const float r12 = fresnel_schlick(cos_theta1, iridescence_ior_to_f0(iri_ior, outside_ior));
    const float t121 = 1.0f - r12;
    const float phi12 = iri_ior < outside_ior ? MERIAN_PI : 0.0f;
    const float phi21 = MERIAN_PI - phi12;

    // Second interface (film | base).
    const float3 base_ior = iridescence_f0_to_ior(clamp(base_f0, 0.0f, 0.9999f));
    const float3 r23 = fresnel_schlick(cos_theta2, iridescence_ior_to_f0(base_ior, iri_ior));
    const float3 phi23 =
        float3(base_ior.x < iri_ior ? MERIAN_PI : 0.0f, base_ior.y < iri_ior ? MERIAN_PI : 0.0f,
               base_ior.z < iri_ior ? MERIAN_PI : 0.0f);

    // Optical path difference and per-interface phase.
    const float opd = 2.0f * iri_ior * thickness * cos_theta2;
    const float3 phi = float3(phi21) + phi23;

    // Airy summation of the reflected orders.
    const float3 r123 = clamp(r12 * r23, 1e-5f, 0.9999f);
    const float3 r123_sqrt = sqrt(r123);
    const float3 rs = pow2(t121) * r23 / (float3(1.0f) - r123);

    float3 result = float3(r12) + rs; // order m = 0
    float3 cm = rs - float3(t121);
    [ForceUnroll] for (int m = 1; m <= 2; ++m) {
        cm *= r123_sqrt;
        const float3 sm = 2.0f * iridescence_sensitivity(float(m) * opd, float(m) * phi);
        result += cm * sm;
    }
    // Clamp to a physical reflectance: the spectral-sensitivity fit can overshoot [0,1] at
    // resonances, which would make the complementary transmittance (1 - R) negative.
    return clamp(result, float3(0.0f), float3(1.0f));
}

} // namespace merian
