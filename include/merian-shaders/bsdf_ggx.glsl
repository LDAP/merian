#ifndef _MERIAN_SHADERS_BSDF_GGX_H_
#define _MERIAN_SHADERS_BSDF_GGX_H_

#include "merian-shaders/common.glsl"
#include "merian-shaders/fresnel.glsl"
#include "merian-shaders/bsdf_diffuse.glsl"
#include "merian-shaders/frames.glsl"

// Smiths shadow masking term
float smith_g1(const float roughness, const float wodotn) {
    const float alpha_sq = pow(roughness, 4);
    return 2.0 * wodotn / (wodotn + sqrt(alpha_sq + (1.0 - alpha_sq) * MERIAN_SQUARE(wodotn)));
}

// minuswi = V
float smith_g_over_minuswidotn(const float roughness, const float minuswidotn, const float wodotn) {
    const float alpha = MERIAN_SQUARE(roughness);
    const float g1 = minuswidotn * sqrt(MERIAN_SQUARE(alpha) + (1.0 - MERIAN_SQUARE(alpha)) * MERIAN_SQUARE(wodotn));
    const float g2 = wodotn * sqrt(MERIAN_SQUARE(alpha) + (1.0 - MERIAN_SQUARE(alpha)) * MERIAN_SQUARE(minuswidotn));
    return 2.0 * wodotn / (g1 + g2);
}

// minuswi = V
vec3 bsdf_ggx_times_wodotn(const vec3 minuswi, const vec3 wo, const vec3 n, const float roughness, const vec3 F0) {
    const vec3 H = normalize(wo + minuswi);

    const float wodotn = clamp(dot(n, wo), 0, 1);
    const float minuswi_dot_h = clamp(dot(minuswi, H), 0, 1);
    const float ndotv = clamp(dot(n, minuswi), 0, 1);
    const float ndoth = clamp(dot(n, H), 0, 1);

    if (wodotn > 0) {
        const float G = smith_g_over_minuswidotn(roughness, ndotv, wodotn);
        const float alpha = MERIAN_SQUARE(roughness);
        const float D = MERIAN_SQUARE(alpha) / (M_PI * MERIAN_SQUARE(MERIAN_SQUARE(ndoth) * MERIAN_SQUARE(alpha) + (1 - MERIAN_SQUARE(ndoth))));

        const vec3 F = fresnel_schlick(minuswi_dot_h, F0);

        return F * (D * G / 4);
    }
    return vec3(0);
}

float bsdf_ggx_times_wodotn(const vec3 minuswi, const vec3 wo, const vec3 n, const float roughness, const float F0) {
    const vec3 H = normalize(wo + minuswi);

    const float wodotn = clamp(dot(n, wo), 0, 1);
    const float minuswi_dot_h = clamp(dot(minuswi, H), 0, 1);
    const float ndotv = clamp(dot(n, minuswi), 0, 1);
    const float ndoth = clamp(dot(n, H), 0, 1);

    if (wodotn > 0) {
        const float G = smith_g_over_minuswidotn(roughness, ndotv, wodotn);
        const float alpha = MERIAN_SQUARE(roughness);
        const float D = MERIAN_SQUARE(alpha) / (M_PI * MERIAN_SQUARE(MERIAN_SQUARE(ndoth) * MERIAN_SQUARE(alpha) + (1 - MERIAN_SQUARE(ndoth))));

        const float F = fresnel_schlick(minuswi_dot_h, F0);

        return F * (D * G / 4);
    }
    return 0;
}

float bsdf_ggx_diffuse_mix_times_wodotn(const vec3 minuswi, const vec3 wo, const vec3 n, const float roughness, const float F0) {
    const vec3 H = normalize(wo + minuswi);

    const float wodotn = clamp(dot(n, wo), 0, 1);
    if (wodotn <= 0) {
        return 0;
    }

    const float minuswi_dot_h = clamp(dot(minuswi, H), 0, 1);
    const float ndotv = clamp(dot(n, minuswi), 0, 1);
    const float ndoth = clamp(dot(n, H), 0, 1);

    const float G = smith_g_over_minuswidotn(roughness, ndotv, wodotn);
    const float alpha = MERIAN_SQUARE(roughness);
    const float D = MERIAN_SQUARE(alpha) / (M_PI * MERIAN_SQUARE(MERIAN_SQUARE(ndoth) * MERIAN_SQUARE(alpha) + (1 - MERIAN_SQUARE(ndoth))));

    const float F = fresnel_schlick(minuswi_dot_h, F0);

    return mix(INV_PI * wodotn, (D * G / 4), F);
    
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
vec3 bsdf_ggx_VNDF_sample_H(const vec2 random, const float roughness, const vec3 Ve) {
    const float alpha = MERIAN_SQUARE(roughness);
    const vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    const float lensq = MERIAN_SQUARE(Vh.x) + MERIAN_SQUARE(Vh.y);
    const vec3 T1 = lensq > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * (1 / sqrt(lensq)) : vec3(1.0, 0.0, 0.0);
    const vec3 T2 = cross(Vh, T1);

    const float r = sqrt(random.x);// sqrt(random.x * ndf_trim);
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

float bsdf_ggx_VNDF_pdf(const float roughness, const vec3 N, const vec3 minus_wi, const vec3 wo) {
    const vec3 H = normalize(wo + minus_wi);
    const float ndoth = clamp(dot(N, H), 0, 1);
    const float minuswi_dot_h = clamp(dot(minus_wi, H), 0, 1);

    const float alpha = MERIAN_SQUARE(roughness);
    const float D = MERIAN_SQUARE(alpha) / (M_PI * MERIAN_SQUARE(MERIAN_SQUARE(ndoth) * MERIAN_SQUARE(alpha) + (1 - MERIAN_SQUARE(ndoth))));
    return (minuswi_dot_h > 0.0) ? D / (4.0 * minuswi_dot_h) : 0.0;
}

float bsdf_ggx_diffuse_mix_pdf(const vec3 minuswi, const vec3 wo, const vec3 n, const float roughness) {
    const float wodotn = dot(wo, n);
    if (wodotn <= 0) {
        return 0;
    }

    const float diffuse_pdf = bsdf_diffuse_pdf(wodotn);
    const float ggx_vndf_pdf = bsdf_ggx_VNDF_pdf(roughness, n, minuswi, wo);

    const float fresnel = fresnel_schlick(dot(minuswi, n), 0.02);
    return mix(diffuse_pdf, ggx_vndf_pdf, fresnel);
}

vec3 bsdf_ggx_diffuse_mix_sample(const vec3 minuswi, const vec3 n, const float roughness, const vec3 random) {
    const float fresnel = fresnel_schlick(dot(minuswi, n), 0.02);

    if (fresnel < random.x) {
        return bsdf_diffuse_sample(n, random.yz);
    } else {
        const mat3x3 frame = make_frame(n);
        const vec3 H = normalize(bsdf_ggx_VNDF_sample_H(random.yz, roughness, world_to_frame(frame, minuswi)));
        return reflect(-minuswi, frame_to_world(frame, H));
    }
}

#endif // _MERIAN_SHADERS_BSDF_GGX_H_
