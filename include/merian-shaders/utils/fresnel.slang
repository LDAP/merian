#pragma once

namespace merian {

/**
 *  F0: Specular reflectance at normal incidence (0 degrees); specular albedo;
 *  pow2((n1-n2)/(n1+n2)).
 *  F90: Specular reflectance at orthogonal incidence (90 degrees); 1.0 for specular reflection.
 *  cos_theta: widoth, cosine of angle between microfacet normal (h) and light
 *  direction (wi).
 */
T fresnel_schlick<T : IFloat>(float cos_theta, T F0, T F90) {
    return (F0) + (F90 - F0) * T(pow(max(1. - (cos_theta), 0), 5.));
}

/**
 *  F0: Specular reflectance at normal incidence (0 degrees); specular albedo;
 *  pow2((n1-n2)/(n1+n2)).
 *  cos_theta: widoth, cosine of angle between microfacet normal (h) and light
 *  direction (wi).
 */
T fresnel_schlick<T : IFloat>(float cos_theta, T F0) {
    return fresnel_schlick(cos_theta, F0, T(1.));
}

}
