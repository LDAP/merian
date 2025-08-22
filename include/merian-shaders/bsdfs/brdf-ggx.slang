#pragma once

#include "merian-shaders/utils/math-constants.slangh"

import "merian-shaders/bsdfs/bsdf.slang";
import "merian-shaders/utils/math.slang";
import "merian-shaders/utils/fresnel.slang";

[shader("compute")]
void dummyMain() {
    return;
}

namespace merian {

/**
 * Converts the roughness (i.e. from Disney BRDF) to GGX alpha.
 */
float ggx_roughness_to_alpha(const float roughness) {
    return pow2(roughness);
}

// -----------------------------------------------------------------

/** Smith lambda function for the GGX normal distribution.
 *  http://jcgt.org/published/0003/02/03/paper.pdf (eq. 72)
 *
 *  wdotn: dot product between shading normal and the evaluated direction, in the positive
 *  hemisphere.
 */
float smith_Lambda(float alpha_squared, float wdotn) {
    if (wdotn <= 0)
        return 0;
    float wdotn_squared = wdotn * wdotn;
    float tan_theta_squared = max(1 - wdotn_squared, 0) / wdotn_squared;
    return 0.5 * (-1 + sqrt(1 + alpha_squared * tan_theta_squared));
}

// Smiths shadow masking term
float smith_g1(const float widotn, const float alpha_squared) {
    return (2.0f * widotn) /
           (widotn + sqrt(alpha_squared + ((1.0f - alpha_squared) * (widotn * widotn))));
}

// Smiths shadow masking term
// Returns G1 / widotn
float smith_g1_over_widotn(const float widotn, const float alpha_squared) {
    return 2.0f / (widotn + sqrt(alpha_squared + ((1.0f - alpha_squared) * (widotn * widotn))));
}

// (Height-correlated) Smiths shadow masking term
//
// http://jcgt.org/published/0003/02/03/paper.pdf (eq. 99)
//
// Returns G2(wodotn, widotn) / widotn
float smith_g2_over_widotn(const float wodotn, const float widotn, const float alpha_squared) {
    float g1 = widotn * sqrt(alpha_squared + ((1.0f - alpha_squared) * (wodotn * wodotn)));
    float g2 = wodotn * sqrt(alpha_squared + ((1.0f - alpha_squared) * (widotn * widotn)));
    return (2.0f * wodotn) / (g1 + g2);
}

// (Height-correlated) Smiths shadow masking term
//
// http://jcgt.org/published/0003/02/03/paper.pdf (eq. 99)
//
// Returns G2(wodotn, widotn) = 1 / (1 + lambda(widotn) + lambda(wodotn));
float smith_g2(const float wodotn, const float widotn, const float alpha_squared) {


    float g1 = widotn * sqrt(alpha_squared + ((1.0f - alpha_squared) * (wodotn * wodotn)));
    float g2 = wodotn * sqrt(alpha_squared + ((1.0f - alpha_squared) * (widotn * widotn)));
    return (2.0f * wodotn * widotn) / (g1 + g2);
}

// -----------------------------------------------------------------
// NDF

// Returns D(h)
float ggx_NDF(float ndoth_squared, float alpha_squared) {
    return alpha_squared /
           (MERIAN_PI * pow2((ndoth_squared * alpha_squared) + (1.0f - ndoth_squared)));
}

/**
 * Samples the GGX NDF using Walter et al. 2007's method.
 *
 * https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
 *
 * Note: The resulting half vector can be in the negative
 */
float3 ggx_NDF_sample_H(float alpha_squared, float2 random) {
    float phi = random.y * (2 * MERIAN_PI);
    float tan_theta_sqr = alpha_squared * random.x / (1 - random.x);
    float cos_theta = 1 / sqrt(1 + tan_theta_sqr);
    float r = sqrt(max(1 - cos_theta * cos_theta, 0));
    return float3(cos(phi) * r, sin(phi) * r, cos_theta);
}

/**
 * The GGX NDF pdf for an half vector sampled using Walter et al. 2007's method. Assumes a local
 * space with n = (0, 0, 1).
 *
 * Returns: D(h) * h.z = D(h) * cos_theta
 */
float ggx_NDF_pdf_H(float3 h, float alpha_squared) {
    return ggx_NDF(pow2(h.z), alpha_squared) * h.z;
}

// -----------------------------------------------------------------
// VNDF

/**
 * Sampling Visible GGX Normals with Spherical Caps
 * Jonathan Dupuy, Anis Benyoub
 * High Performance Graphics 2023
 *
 * Samples a half vector h.
 *
 * https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0
 */
float3 ggx_VNDF_sample_H(float3 wi, float2 alpha, float2 random) {
    // warp to the hemisphere configuration
    float3 wiStd = normalize(float3(wi.xy * alpha, wi.z));
    // sample a spherical cap in (-wi.z, 1]
    float phi = ((2.0f * random.x) - 1.0f) * MERIAN_PI;
    float z = mad(1.0f - random.y, 1.0f + wiStd.z, -wiStd.z);
    float sinTheta = sqrt(clamp(1.0f - (z * z), 0.0f, 1.0f));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float3 c = float3(x, y, z);
    // compute halfway direction as standard normal
    float3 wmStd = c + wiStd;
    // warp back to the ellipsoid configuration
    float3 wm = normalize(float3(wmStd.xy * alpha, wmStd.z));
    return wm;
}

/**
 * Sampling Visible GGX Normals with Spherical Caps
 * Jonathan Dupuy, Anis Benyoub
 * High Performance Graphics 2023
 *
 * Samples a half vector h.
 *
 * https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0
 */
float3 ggx_VNDF_sample_H(float3 wi, float3 n, float alpha, float2 random) {
    // decompose the vector in parallel and perpendicular components
    float3 wi_z = n * dot(wi, n);
    float3 wi_xy = wi - wi_z;
    // warp to the hemisphere configuration
    float3 wiStd = normalize(wi_z - (wi_xy * alpha));
    // sample a spherical cap in (-wiStd.z, 1]
    float wiStd_z = dot(wiStd, n);
    float phi = ((2.0f * random.x) - 1.0f) * MERIAN_PI;
    float z = ((1.0f - random.y) * (1.0f + wiStd_z)) - wiStd_z;
    float sinTheta = sqrt(clamp(1.0f - (z * z), 0.0f, 1.0f));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float3 cStd = float3(x, y, z);
    // reflect sample to align with normal
    // @LDAP: Changed to 1.0000001 to prevent dividing by 0.
    float3 wr = n + float3(0.0f, 0.0f, 1.0000001);
    float3 c = ((wr * dot(wr, cStd)) / wr.z.xxx) - cStd;
    // compute halfway direction as standard normal
    float3 wmStd = c + wiStd;
    float3 wmStd_z = n * dot(n, wmStd);
    float3 wmStd_xy = wmStd_z - wmStd;
    // warp back to the ellipsoid configuration
    float3 wm = normalize(wmStd_z + (wmStd_xy * alpha));
    return wm;
}

/**
 * Sampling Visible GGX Normals with Spherical Caps
 * Jonathan Dupuy, Anis Benyoub
 * High Performance Graphics 2023
 *
 * Samples an outgoing vector wo.
 *
 * https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0
 */
float3 ggx_VNDF_sample_wo(float3 wi, float3 n, float alpha, float2 random) {
    float3 h = ggx_VNDF_sample_H(wi, n, alpha, random);
    return reflect2(wi, h);
}

/**
 * Sampling Visible GGX Normals with Spherical Caps
 * Jonathan Dupuy, Anis Benyoub
 * High Performance Graphics 2023
 *
 * Samples an outgoing vector wo.
 *
 * https://gist.github.com/jdupuy/4c6e782b62c92b9cb3d13fbb0a5bd7a0
 */
float3 ggx_VNDF_sample_wo(float3 wi, float alpha, float2 random) {
    float3 h = ggx_VNDF_sample_H(wi, alpha, random);
    return reflect2(wi, h);
}

/**
 * The VNDF pdf for an half vector.
 *
 * Returns: G1(wi) * D(h) * max(0,dot(wi,h)) / wodotn
 */
float ggx_VNDF_pdf_H(float widotn, float widoth, float ndoth_squared, float alpha_squared) {
    float G1_over_widotn = smith_g1_over_widotn(widotn, alpha_squared);
    float D = ggx_NDF(ndoth_squared, alpha_squared);
    return D * G1_over_widotn * max(0.f, widoth);
}

/**
 * The VNDF pdf for an half vector. Assumes a local space with n = (0, 0, 1).
 *
 * Returns: G1(wi) * D(h) * max(0,dot(wi,h)) / wi.z
 */
float ggx_VNDF_H_pdf(float3 wi, float3 h, float alpha_squared) {
    float G1_over_widotn = smith_g1_over_widotn(wi.z, alpha_squared);
    float D = ggx_NDF(pow2(h.z), alpha_squared);
    /* / (4 * dot(wi, h)) is the Jacobian of the reflection */
    return D * G1_over_widotn * max(0.f, dot(wi, h));
}

/**
 * The VNDF pdf for an outgoing direction (has the Jacobian of the reflection applied). Assumes that
 * widoth > 0.
 *
 */
float ggx_VNDF_pdf_wo(float widotn, float ndoth_squared, float alpha_squared) {
    float G1_over_widotn = smith_g1_over_widotn(widotn, alpha_squared);
    float D = ggx_NDF(ndoth_squared, alpha_squared);
    /* / (4 * dot(wi, h)) is the Jacobian of the reflection */
    return (D * G1_over_widotn /* * max(0.f, dot(wi, h)*/) / 4.0f /* dot(wi, h) */;
}

/**
 * The VNDF pdf for an outgoing direction (has the Jacobian of the reflection applied).
 */
float ggx_VNDF_pdf_wo(float3 wi, float3 wo, float3 n, float alpha_squared) {
    float3 h = normalize(wo + wi);
    float widotn = dot(wi, n);
    float ndoth = max(0.0f, dot(n, h));
    return ggx_VNDF_pdf_wo(widotn, pow2(ndoth), alpha_squared);
}

/**
 * The VNDF pdf for an outgoing direction (has the Jacobian of the reflection applied). Assumes a
 * local space with n = (0, 0, 1).
 */
float ggx_VNDF_pdf_wo(float3 wi, float3 wo, float alpha_squared) {
    float3 h = normalize(wo + wi);
    return ggx_VNDF_pdf_wo(wi.z, pow2(max(0.f, h.z)), alpha_squared);
}

// -----------------------------------------------------------------

// float ggx_times_wodotn(float wodotn, float widotn, float ndoth, float alpha) {
//     float G2 = smith_g2_over_widotn(wodotn, widotn, alpha * alpha);
//     float D = ggx_NDF(ndoth * ndoth, alpha * alpha);
//     return (D * G2) / 4.0f;
// }
//
// float ggx_times_wodotn(float3 wi, float3 wo, float3 n, float alpha) {
//     float3 h = normalize(wo + wi);
//     float wodotn = dot(wo, n);
//     float widotn = dot(wi, n);
//     float ndoth = dot(n, h);
//     return ggx_times_wodotn(wodotn, widotn, ndoth, alpha);
// }
//
// float ggx_times_wodotn(float3 wi, float3 wo, float3 n, float alpha, float F0) {
//     float3 h = normalize(wo + wi);
//     float wodotn = dot(wo, n);
//     float widotn = dot(wi, n);
//     float ndoth = dot(n, h);
//     float bsdf = ggx_times_wodotn(wodotn, widotn, ndoth, alpha);
//     float widoth = dot(wi, h);
//     float F = F0 + ((1.0f - F0) * pow(max(1.0f - widoth, 0.0f), 5.0f));
//     return F * bsdf;
// }
//
// float3 ggx_times_wodotn(float3 wi, float3 wo, float3 n, float alpha, float3 F0) {
//     float3 h = normalize(wo + wi);
//     float wodotn = dot(wo, n);
//     float widotn = dot(wi, n);
//     float ndoth = dot(n, h);
//     float bsdf = ggx_times_wodotn(wodotn, widotn, ndoth, alpha);
//     float widoth = dot(wi, h);
//     float3 F = F0 + ((1.0f.xxx - F0) * pow(max(1.0f - widoth, 0.0f), 5.0f));
//     return F * bsdf;
// }

// A specular microfacet BRDF with GGX normal distribution and (Schlick-)Fresnel.
struct GGXBRDF : BSDF {
    float3 albedo;
    float alpha;

    __init(float3 albedo, float alpha) {
        this.albedo = albedo;
        this.alpha = alpha;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return float3(0.0f);
        }

        const float3 h = normalize(wo + wi);
        const float alpha_squared = pow2(alpha);
        float G2_over_widotn =
            smith_g2_over_widotn(wo.z /*wodotn*/, wi.z /*widotn*/, alpha_squared);
        float D = ggx_NDF(pow2(h.z), alpha_squared);

        const float widoth = dot(wi, h);
        const float3 F = fresnel_schlick(widoth, albedo);

        return (F * D * G2_over_widotn) / 4.0f;
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return none;
        }

        float3 h = ggx_VNDF_sample_H(wi, alpha, rng.next_float2());
        float widoth = dot(wi, h);
        float3 wo = reflect2(wi, h, widoth);

        if (wo.z < BSDF_MIN_COS_THETA) {
            return none;
        }

        return wo;
    }

    // TODO: Is this correct? (weight computation...)
    // override Optional<BSDFSample> sample_eval(const float3 wi, inout RandomGenerator rng) {
    //     if (wi.z < BSDF_MIN_COS_THETA) {
    //         return none;
    //     }

    //     BSDFSample s;

    //     float3 h = ggx_VNDF_sample_H(wi, alpha, rng.next_float2());
    //     float widoth = dot(wi, h);
    //     s.wo = reflect2(wi, h, widoth);

    //     if (s.wo.z < BSDF_MIN_COS_THETA) {
    //         return none;
    //     }

    //     s.pdf = ggx_VNDF_pdf_wo(wi.z, pow2(max(0.f, h.z)), pow2(alpha));

    //     float G = smith_g2(s.wo.z, wi.z, pow2(alpha));
    //     float G_over_G1wo = G * (1.f + smith_Lambda(pow2(alpha), wi.z));
    //     const float3 F = fresnel_schlick(widoth, albedo);

    //     s.weight = F * G_over_G1wo;

    //     return s;
    // }

    float pdf(const float3 wi, const float3 wo) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return 0.0f;
        }

        float3 h = normalize(wi + wo);
        return ggx_VNDF_pdf_wo(wi.z, pow2(max(0.f, h.z)), pow2(alpha));
    }

    Albedo get_albedo(const float3 wi) {
        const float3 F = fresnel_schlick(wi.z, albedo);
        return Albedo(F);
    }
}

// float ggx_diffuse_mix_times_wodotn(float3 wi, float3 wo, float3 n, float alpha, float F0) {
//     float3 h = normalize(wo + wi);
//     float wodotn = clamp(dot(n, wo), 0.0f, 1.0f);
//     if (wodotn <= 0.0f) {
//         return 0.0f;
//     }
//     float widotn = dot(wi, n);
//     float ndoth = dot(n, h);
//     float widoth = dot(wi, h);
//     float F = F0 + ((1.0f - F0) * pow(max(1.0f - widoth, 0.0f), 5.0f));
//     return lerp(0.3183098733425140380859375f * wodotn,
//                 ggx_times_wodotn(wodotn, widotn, ndoth, alpha), F);
// }

// float ggx_diffuse_mix_pdf(float3 wi, float3 wo, float3 n, float alpha) {
//     float wodotn = dot(wo, n);
//     if (wodotn <= 0.0f) {
//         return 0.0f;
//     }
//     float3 h = normalize(wo + wi);
//     float widotn = dot(wi, n);
//     float ndoth = clamp(dot(n, h), 0.0f, 1.0f);
//     float diffuse_pdf = 0.3183098733425140380859375f * wodotn;
//     float ggx_vndf_pdf = ggx_VNDF_pdf(widotn, ndoth * ndoth, alpha * alpha);
//     float fresnel = 0.0199999995529651641845703125f +
//                     (0.980000019073486328125f * pow(max(1.0f - widotn, 0.0f), 5.0f));
//     return lerp(diffuse_pdf, ggx_vndf_pdf, fresnel);
// }

// float3 ggx_diffuse_mix_sample(float3 wi, float3 n, float alpha, float3 random) {
//     float fresnel = 0.0199999995529651641845703125f +
//                     (0.980000019073486328125f * pow(max(1.0f - dot(wi, n), 0.0f), 5.0f));
//     if (fresnel < random.x) {
//         return mul(sample_cos(random.yz), make_frame(n));
//     } else {
//         return ggx_VNDF_sample(wi, n, alpha, random.yz);
//     }
// }

}
