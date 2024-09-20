#ifndef _MERIAN_SHADERS_BSDF_GGX_H_
#define _MERIAN_SHADERS_BSDF_GGX_H_

#include "merian-shaders/common.glsl"

// Smiths shadow masking term
float smith_g1(const float roughness, const float wodotn) {
    const float alpha_sq = pow(roughness, 4);
    return 2.0 * wodotn / (wodotn + sqrt(alpha_sq + (1.0 - alpha_sq) * MERIAN_SQUARE(wodotn)));
}

// minuswi = V
float smith_g_over_minuswidotn(const float roughness, const float minuswidotn, const float wodotn) {
    float alpha = MERIAN_SQUARE(roughness);
    float g1 = minuswidotn * sqrt(MERIAN_SQUARE(alpha) + (1.0 - MERIAN_SQUARE(alpha)) * MERIAN_SQUARE(wodotn));
    float g2 = wodotn * sqrt(MERIAN_SQUARE(alpha) + (1.0 - MERIAN_SQUARE(alpha)) * MERIAN_SQUARE(minuswidotn));
    return 2.0 * wodotn / (g1 + g2);
}

// returns a half vector in tangent space
vec3 bsdf_ggx_sample_H(const vec2 random, const float roughness) {
    const float phi = TWO_PI * random.x;
    const float cosTheta = sqrt((1 - random.y) / (1 + (pow(roughness, 4) - 1) * random.y));
    const float sinTheta = sqrt(1 - cosTheta * cosTheta);

    return vec3(sinTheta * cos(phi), 
                sinTheta * sin(phi),
                cosTheta);
}

// https://github.com/NVIDIAGameWorks/donut
// MIT Licensed
vec3 bsdf_ggx_sample_H_VNDF(const vec3 random, const float roughness, const vec3 Ve, const float ndf_trim) {
    const float alpha = MERIAN_SQUARE(roughness);
    const vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    const float lensq = MERIAN_SQUARE(Vh.x) + MERIAN_SQUARE(Vh.y);
    const vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * (1 / sqrt(lensq)) : vec3(1.0, 0.0, 0.0);
    const vec3 T2 = cross(Vh, T1);

    const float r = sqrt(random.x * ndf_trim);
    const float phi = 2.0 * M_PI * random.y;
    const float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    const float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - MERIAN_SQUARE(t1)) + s * t2;

    const vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - MERIAN_SQUARE(t1) - MERIAN_SQUARE(t2))) * Vh;

    return vec3(alpha * Nh.x,
                alpha * Nh.y,
                max(0.0, Nh.z)
        );
}

float bsdf_ggx_sample_VNDF_PDF(const float roughness, const vec3 N, const vec3 minus_wi, const vec3 wo) {
    vec3 H = normalize(wo + minus_wi);
    float ndoth = clamp(dot(N, H), 0, 1);
    float minus_wi_dot_h = clamp(dot(minus_wi, H), 0, 1);

    float alpha = MERIAN_SQUARE(roughness);
    float D = MERIAN_SQUARE(alpha) / (M_PI * MERIAN_SQUARE(MERIAN_SQUARE(ndoth) * MERIAN_SQUARE(alpha) + (1 - MERIAN_SQUARE(ndoth))));
    return (minus_wi_dot_h > 0.0) ? D / (4.0 * minus_wi_dot_h) : 0.0;
}

#endif // _MERIAN_SHADERS_BSDF_GGX_H_
