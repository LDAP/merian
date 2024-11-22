#ifndef _MERIAN_SHADERS_BSDF_GGX_H_
#define _MERIAN_SHADERS_BSDF_GGX_H_

#include "merian-shaders/common.glsl"
#include "merian-shaders/fresnel.glsl"
#include "merian-shaders/bsdf_diffuse.glsl"
#include "merian-shaders/frames.glsl"

float roughness_to_alpha(const float roughness) {
    return MERIAN_SQUARE(roughness);
}

// ----------------------------------------

float bsdf_ggx_G1(const vec3 wo, const vec3 n, const float alpha_squared) {
    const float NdotL = dot(wo, n);
    return 2.0 * NdotL / (NdotL + sqrt(alpha_squared + (1.0 - alpha_squared) * MERIAN_SQUARE(NdotL)));
}

// GGX normal distribution function
float bsdf_ggx_D(float roughness, const vec3 n, const vec3 h) {
  float cos2 = dot(n, h)*dot(n, h);  // cos2 theta
  float sin2 = max(0.0, 1.0-cos2);
  float a = sin2 / cos2 + roughness * roughness;
  float k = roughness / (cos2 * a);
  return k * k / M_PI;
}

// ----------------------------------------

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

/**
 * Sampling Visible GGX Normals with Spherical Caps
 * Jonathan Dupuy, Anis Benyoub
 * High Performance Graphics 2023
 * 
 * https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0
 */
vec3 bsdf_ggx_VNDF_H_sample(const vec3 minuswi, const vec3 n, const float alpha, const vec2 random) {
    // decompose the vector in parallel and perpendicular components
    const vec3 wi_z = n * dot(minuswi, n);
    const vec3 wi_xy = minuswi - wi_z;
    // warp to the hemisphere configuration
    const vec3 wiStd = normalize(wi_z - alpha * wi_xy);
    // sample a spherical cap in (-wiStd.z, 1]
    const float wiStd_z = dot(wiStd, n);
    const float phi = (2.0f * random.x - 1.0f) * M_PI;
    const float z = (1.0f - random.y) * (1.0f + wiStd_z) - wiStd_z;
    const float sinTheta = sqrt(clamp(1.0f - z * z, 0.0f, 1.0f));
    const float x = sinTheta * cos(phi);
    const float y = sinTheta * sin(phi);
    const vec3 cStd = vec3(x, y, z);
    // reflect sample to align with normal
    const vec3 up = vec3(0, 0, 1);
    const vec3 wr = n + up;
    const vec3 c = dot(wr, cStd) * wr / wr.z - cStd;
    // compute halfway direction as standard normal
    const vec3 wmStd = c + wiStd;
    const vec3 wmStd_z = n * dot(n, wmStd);
    const vec3 wmStd_xy = wmStd_z - wmStd;
    // warp back to the ellipsoid configuration
    const vec3 wm = normalize(wmStd_z + alpha * wmStd_xy);
    // return final normal
    return wm;
}

// -------------------------------------------

// (solid angle, contains multiplication with wodotn)
float bsdf_ggx_VNDF_pdf(const vec3 wi, const vec3 wo, const vec3 n, const float alpha) {
    const vec3 h = normalize(wo - wi);
    const float D  = bsdf_ggx_D(alpha, n, h);
    const float G1 = bsdf_ggx_G1(wo, n, alpha * alpha);
    return (G1 * dot(-wi, h) * D) / (dot(-wi, n) * 4) * dot(wo, n);
}

vec3 bsdf_ggx_VNDF_sample(const vec3 wi, const vec3 n, const float alpha, const vec2 random) {
    const vec3 h = bsdf_ggx_VNDF_H_sample(-wi, n, alpha, random);
    return reflect(wi, h);
}

// -------------------------------------------

// float bsdf_ggx_diffuse_mix_times_wodotn(const vec3 minuswi, const vec3 wo, const vec3 n, const float roughness, const float F0) {
//     const vec3 H = normalize(wo + minuswi);

//     const float wodotn = clamp(dot(n, wo), 0, 1);
//     if (wodotn <= 0) {
//         return 0;
//     }

//     const float minuswi_dot_h = clamp(dot(minuswi, H), 0, 1);
//     const float ndotv = clamp(dot(n, minuswi), 0, 1);
//     const float ndoth = clamp(dot(n, H), 0, 1);

//     const float G = smith_g_over_minuswidotn(roughness, ndotv, wodotn);
//     const float alpha = MERIAN_SQUARE(roughness);
//     const float D = MERIAN_SQUARE(alpha) / (M_PI * MERIAN_SQUARE(MERIAN_SQUARE(ndoth) * MERIAN_SQUARE(alpha) + (1 - MERIAN_SQUARE(ndoth))));

//     const float F = fresnel_schlick(minuswi_dot_h, F0);

//     return mix(INV_PI * wodotn, (D * G / 4), F);
    
// }

// float bsdf_ggx_diffuse_mix_pdf(const vec3 minuswi, const vec3 wo, const vec3 n, const float roughness) {
//     const float wodotn = dot(wo, n);
//     if (wodotn <= 0) {
//         return 0;
//     }

//     const float diffuse_pdf = bsdf_diffuse_pdf(wodotn);
//     const float ggx_vndf_pdf = bsdf_ggx_VNDF_pdf(minuswi, wo, n, roughness_to_alpha(roughness));

//     const float fresnel = fresnel_schlick(dot(minuswi, n), 0.02);
//     return mix(diffuse_pdf, ggx_vndf_pdf, fresnel);
// }

// vec3 bsdf_ggx_diffuse_mix_sample(const vec3 minuswi, const vec3 n, const float roughness, const vec3 random) {
//     const float fresnel = fresnel_schlick(dot(minuswi, n), 0.02);

//     if (fresnel < random.x) {
//         return bsdf_diffuse_sample(n, random.yz);
//     } else {
//         const mat3x3 frame = make_frame(n);
//         const vec3 H = normalize(bsdf_ggx_VNDF_H_sample(random.yz, roughness, world_to_frame(frame, minuswi)));
//         return reflect(-minuswi, frame_to_world(frame, H));
//     }
// }

#endif // _MERIAN_SHADERS_BSDF_GGX_H_
