#ifndef _MERIAN_SHADERS_BSDF_GGX_H_
#define _MERIAN_SHADERS_BSDF_GGX_H_

#include "merian-shaders/common.glsl"
#include "merian-shaders/fresnel.glsl"
#include "merian-shaders/bsdf_diffuse.glsl"
#include "merian-shaders/frames.glsl"

// -----------------------------------------------------------------

float bsdf_ggx_roughness_to_alpha(const float roughness) {
    return roughness * roughness;
}

// Smiths shadow masking term
float bsdf_ggx_smith_g1(const float minuswidotn, const float alpha_squared) {
    return 2.0 * minuswidotn / (minuswidotn + sqrt(alpha_squared + (1.0 - alpha_squared) * MERIAN_SQUARE(minuswidotn)));
}

// Smiths shadow masking term
float bsdf_ggx_smith_g1_over_minuswidotn(const float minuswidotn, const float alpha_squared) {
    return 2.0 / (minuswidotn + sqrt(alpha_squared + (1.0 - alpha_squared) * MERIAN_SQUARE(minuswidotn)));
}

// minuswi = V
float bsdf_ggx_smith_g2_over_minuswidotn(const float wodotn, const float minuswidotn, const float alpha_squared) {
    const float g1 = minuswidotn * sqrt(alpha_squared + (1.0 - alpha_squared) * MERIAN_SQUARE(wodotn));
    const float g2 = wodotn * sqrt(alpha_squared + (1.0 - alpha_squared) * MERIAN_SQUARE(minuswidotn));
    return 2.0 * wodotn / (g1 + g2);
}

float bsdf_ggx_D(const float ndoth_squared, const float alpha_squared) {
    return alpha_squared / (M_PI * merian_square(ndoth_squared * alpha_squared + (1 - ndoth_squared)));
}

// -----------------------------------------------------------------

float bsdf_ggx_times_wodotn(const float wodotn, const float minuswidotn, const float ndoth, const float alpha) {
    const float G2 = bsdf_ggx_smith_g2_over_minuswidotn(wodotn, minuswidotn, alpha * alpha);
    const float D = bsdf_ggx_D(ndoth * ndoth, alpha * alpha);

    return (D * G2) / 4;
}


float bsdf_ggx_times_wodotn(const vec3 wi, const vec3 wo, const vec3 n, const float alpha) {
    const vec3 h = normalize(wo - wi);

    const float wodotn = dot(wo, n);
    const float minuswidotn = dot(-wi, n);
    const float ndoth = dot(n, h);

    return bsdf_ggx_times_wodotn(wodotn, minuswidotn, ndoth, alpha);
}

// with fresnel
float bsdf_ggx_times_wodotn(const vec3 wi, const vec3 wo, const vec3 n, const float alpha, const float F0) {
    const vec3 h = normalize(wo - wi);

    const float wodotn = dot(wo, n);
    const float minuswidotn = dot(-wi, n);
    const float ndoth = dot(n, h);

    const float bsdf = bsdf_ggx_times_wodotn(wodotn, minuswidotn, ndoth, alpha);
    const float minuswidoth = dot(-wi, h);

    const float F = fresnel_schlick(minuswidoth, F0);

    return F * bsdf;
}

// with fresnel
vec3 bsdf_ggx_times_wodotn(const vec3 wi, const vec3 wo, const vec3 n, const float alpha, const vec3 F0) {
    const vec3 h = normalize(wo - wi);

    const float wodotn = dot(wo, n);
    const float minuswidotn = dot(-wi, n);
    const float ndoth = dot(n, h);

    const float bsdf = bsdf_ggx_times_wodotn(wodotn, minuswidotn, ndoth, alpha);
    const float minuswidoth = dot(-wi, h);

    const vec3 F = fresnel_schlick(minuswidoth, F0);

    return F * bsdf;
}


// -----------------------------------------------------------------

float bsdf_ggx_diffuse_mix_times_wodotn(const vec3 wi, const vec3 wo, const vec3 n, const float alpha, const float F0) {
    const vec3 h = normalize(wo - wi);

    const float wodotn = clamp(dot(n, wo), 0, 1);
    if (wodotn <= 0) {
        return 0;
    }

    const float minuswidotn = dot(-wi, n);
    const float ndoth = dot(n, h);
    const float minuswidoth = dot(-wi, h);

    const float F = fresnel_schlick(minuswidoth, F0);

    return mix(bsdf_diffuse() * wodotn,
               bsdf_ggx_times_wodotn(wodotn, minuswidotn, ndoth, alpha),
               F);
    
}

/**
 * Sampling Visible GGX Normals with Spherical Caps
 * Jonathan Dupuy, Anis Benyoub
 * High Performance Graphics 2023
 * 
 * https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0
 */
vec3 bsdf_ggx_VNDF_H_sample(const vec3 minuswi, const vec2 alpha, const vec2 random) {
    // warp to the hemisphere configuration
    vec3 wiStd = normalize(vec3(minuswi.xy * alpha, minuswi.z));
    // sample a spherical cap in (-minuswi.z, 1]
    float phi = (2.0f * random.x - 1.0f) * M_PI;
    float z = fma((1.0f - random.y), (1.0f + wiStd.z), -wiStd.z);
    float sinTheta = sqrt(clamp(1.0f - z * z, 0.0f, 1.0f));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    vec3 c = vec3(x, y, z);
    // compute halfway direction as standard normal
    vec3 wmStd = c + wiStd;
    // warp back to the ellipsoid configuration
    vec3 wm = normalize(vec3(wmStd.xy * alpha, wmStd.z));
    // return final normal
    return wm;
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

vec3 bsdf_ggx_VNDF_sample(const vec3 wi, const vec3 n, const float alpha, const vec2 random) {
    const vec3 h = bsdf_ggx_VNDF_H_sample(-wi, n, alpha, random);
    return reflect(wi, h);
}

float bsdf_ggx_VNDF_pdf(const vec3 wi, const vec3 wo, const vec3 n, const float alpha) {
    const vec3 h = normalize(wo - wi);
    const float minuswidotn = dot(-wi, n);
    const float ndoth = clamp(dot(n, h), 0, 1);

    const float G1 = bsdf_ggx_smith_g1_over_minuswidotn(minuswidotn, alpha * alpha);
    const float D = bsdf_ggx_D(ndoth * ndoth, alpha * alpha);

    return (D * G1) / 4;
}

float bsdf_ggx_diffuse_mix_pdf(const vec3 wi, const vec3 wo, const vec3 n, const float alpha) {
    const float wodotn = dot(wo, n);

    if (wodotn <= 0) {
        return 0;
    }

    const float diffuse_pdf = bsdf_diffuse_pdf(wodotn);
    const float ggx_vndf_pdf = bsdf_ggx_VNDF_pdf(wi, wo, n, alpha);

    const float fresnel = fresnel_schlick(dot(-wi, n), 0.02);
    return mix(diffuse_pdf, ggx_vndf_pdf, fresnel);
}

vec3 bsdf_ggx_diffuse_mix_sample(const vec3 wi, const vec3 n, const float alpha, const vec3 random) {
    const float fresnel = fresnel_schlick(dot(-wi, n), 0.02);

    if (fresnel < random.x) {
        return bsdf_diffuse_sample(n, random.yz);
    } else {
        return bsdf_ggx_VNDF_sample(wi, n, alpha, random.yz);
    }
}

#endif // _MERIAN_SHADERS_BSDF_GGX_H_
