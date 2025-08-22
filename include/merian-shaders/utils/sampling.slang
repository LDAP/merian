#include "math-constants.slangh"

import "merian-shaders/utils/random.slang";
import "merian-shaders/utils/frame.slang";

namespace merian {

// sample sphere, p = 1 / 4pi
float3 sample_sphere(float2 random) {
    float z = 2.0f * (random.x - 0.5f);
    float z2 = sqrt(1.0f - (z * z));
    return float3(z2 * cos(MERIAN_TWO_PI * random.y), z2 * sin(MERIAN_TWO_PI * random.y), z);
}

// sample hemisphere (around 0,0,1), p = 1 / 2pi
float3 sample_hemisphere(float2 random) {
    float su = sqrt(1.0f - ((1.0f - random.x) * (1.0f - random.x)));
    return float3(su * cos(MERIAN_TWO_PI * random.y), su * sin(MERIAN_TWO_PI * random.y),
                  1.0f - random.x);
}

// sample hemisphere (around 0,0,1), cos lobe, p = cos(theta) / pi
float3 sample_cos(float2 random) {
    float su = sqrt(random.x);
    return float3(su * cos(MERIAN_TWO_PI * random.y), su * sin(MERIAN_TWO_PI * random.y),
                  sqrt(1.0f - random.x));
}

// samples barycentric coordinates of a triangle, p = 1 / A
float3 sample_triangle(float2 random) {
    float x = sqrt(random.x);
    return float3(1.0f - x, x * (1.0f - random.y), x * random.y);
}

// samples coordinates on a unit disk, p = 1 / A
float2 sample_disk(float2 random) {
    float angle = MERIAN_TWO_PI * random.x;
    return float2(cos(angle), sin(angle)) * sqrt(random.y);
}

float2 sample_normal_box_muller(float mean, float sigma, float2 random) {
    return (float2(cos(MERIAN_TWO_PI * random.y), sin(MERIAN_TWO_PI * random.y)) *
            (sigma * sqrt((-2.0f) * log(1.0f - random.x)))) +
           mean.xx;
}

/**
 * This samples a vector on the cone around w with angle cos_theta.
 *
 * i.e for a paricle traveling in direction w, cos_theta is the deflection.
 */
float3 sample_cone(const float3 w, const float cos_theta, const float xi) {
    const float sin_theta = sqrt(max(0., 1.0 - cos_theta * cos_theta));

    float3 tangent, bitangent;
    make_frame(w, tangent, bitangent);

    const float phi = MERIAN_TWO_PI * xi;
    return tangent * sin_theta * cos(phi) + bitangent * sin_theta * sin(phi) + w * cos_theta;
}

/**
 * This samples a vector on the cone around -w with angle cos_theta.
 *
 * i.e for a particle coming from direction w, cos_theta is the deflection (from -w).
 */
float3 sample_cone2(const float3 w, const float cos_theta, const float xi) {
    const float sin_theta = sqrt(max(0., 1.0 - cos_theta * cos_theta));

    float3 tangent, bitangent;
    make_frame(w, tangent, bitangent);

    const float phi = MERIAN_TWO_PI * xi;
    return tangent * sin_theta * cos(phi) + bitangent * sin_theta * sin(phi) - w * cos_theta;
}

float sample_normal_marsaglia(float mean,
                              float sigma,
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

}
