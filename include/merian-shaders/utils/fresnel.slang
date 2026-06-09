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
 * Unpolarized Fresnel reflectance at a dielectric interface. eta is the relative index of refraction
 * (incident / transmitted), cos_i is the cosine of the incidence angle (>= 0). Returns 1.0 on total
 * internal reflection.
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

}
