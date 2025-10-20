#include "math-constants.slangh"

import "merian-shaders/utils/random.slang";
import "merian-shaders/utils/frame.slang";

namespace merian {

// sample sphere, p = 1 / 4pi
float3 sample_sphere(inout RandomGenerator rng) {
    const float z = 2.0f * (rng.next_float() - 0.5f);
    const float z2 = sqrt(1.0f - (z * z));
    const float xi = rng.next_float();
    return float3(z2 * cos(MERIAN_TWO_PI * xi), z2 * sin(MERIAN_TWO_PI * xi), z);
}

// sample hemisphere (around 0,0,1), p = 1 / 2pi
float3 sample_hemisphere(inout RandomGenerator rng) {
    const float2 xi = rng.next_float2();
    const float su = sqrt(1.0f - ((1.0f - xi.x) * (1.0f - xi.x)));
    return float3(su * cos(MERIAN_TWO_PI * xi.y), su * sin(MERIAN_TWO_PI * xi.y), 1.0f - xi.x);
}

// sample hemisphere (around 0,0,1), cos lobe, p = cos(theta) / pi
float3 sample_cos(inout RandomGenerator rng) {
    const float2 xi = rng.next_float2();
    const float su = sqrt(xi.x);
    return float3(su * cos(MERIAN_TWO_PI * xi.y), su * sin(MERIAN_TWO_PI * xi.y),
                  sqrt(1.0f - xi.x));
}

// samples barycentric coordinates of a triangle, p = 1 / A
float3 sample_triangle(inout RandomGenerator rng) {
    const float x = sqrt(rng.next_float());
    const float xi = rng.next_float();
    return float3(1.0f - x, x * (1.0f - xi), x * xi);
}

// samples coordinates on a unit disk, p = 1 / A
float2 sample_disk(inout RandomGenerator rng) {
    const float angle = MERIAN_TWO_PI * rng.next_float();
    return float2(cos(angle), sin(angle)) * sqrt(rng.next_float());
}

// returns two samples!
float2 sample_normal_box_muller(const float mean, const float sigma, inout RandomGenerator rng) {
    float xi = rng.next_float();
    return (float2(cos(MERIAN_TWO_PI * xi), sin(MERIAN_TWO_PI * xi)) *
            (sigma * sqrt((-2.0f) * log(1.0f - rng.next_float())))) +
           mean;
}

float sample_normal_marsaglia(const float mean,
                              const float sigma,
                              inout RandomGenerator rng,
                              inout float spare) {
    if (!isnan(spare)) {
        float res = (spare * sigma) + mean;
        spare = asfloat(0x7fc00000u /* nan */);
        return res;
    } else {
        float2 u;
        float s;
        int i = 0;
        do {
            u = rng.next_float() * 2.0 - 1.0;
            s = u.x * u.x + u.y * u.y;
        } while ((s >= 1.0 || s == 0.0) && i++ < 10);
        s = sqrt(-2.0 * log(s) / s);
        spare = u.y * s;
        return u.x * s * sigma + mean;
    }
}

float sample_normal_pdf(float mean, float sigma, float x) {
    float z = (x - mean) / sigma;
    return (MERIAN_INV_SQRT_TWO_PI * exp(((-0.5f) * z) * z)) / sigma;
}

/**
 * This samples a vector on the cone around w with angle cos_theta.
 *
 * i.e for a paricle traveling in direction w, cos_theta is the deflection.
 */
float3 sample_cone(const float3 w, const float cos_theta, inout RandomGenerator rng) {
    const float sin_theta = sqrt(max(0., 1.0 - cos_theta * cos_theta));

    float3 tangent, bitangent;
    make_frame(w, tangent, bitangent);

    const float phi = MERIAN_TWO_PI * rng.next_float();
    return tangent * sin_theta * cos(phi) + bitangent * sin_theta * sin(phi) + w * cos_theta;
}

/**
 * This samples a vector on the cone around -w with angle cos_theta.
 *
 * i.e for a particle coming from direction w, cos_theta is the deflection (from -w).
 */
float3 sample_cone2(const float3 w, const float cos_theta, inout RandomGenerator rng) {
    const float sin_theta = sqrt(max(0., 1.0 - cos_theta * cos_theta));

    float3 tangent, bitangent;
    make_frame(w, tangent, bitangent);

    const float phi = MERIAN_TWO_PI * rng.next_float();
    return tangent * sin_theta * cos(phi) + bitangent * sin_theta * sin(phi) - w * cos_theta;
}

}
