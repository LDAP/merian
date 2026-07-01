#include "merian-shaders/utils/math-constants.slangh"

import bsdf;
import merian_shaders.utils.math;
import merian_shaders.utils.fresnel;

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

// Smiths shadow masking term
// Returns G1 / wi.z
float smith_g1_over_widotn(const float3 wi, const float2 alpha) {
    return 2.0f / (wi.z + length(float3(alpha * wi.xy, wi.z)));
}

// (Height-correlated) Smiths shadow masking term
//
// Returns G2(wo, wi) / wi.z
float smith_g2_over_widotn(const float3 wo, const float3 wi, const float2 alpha) {
    const float g1 = wi.z * length(float3(alpha * wo.xy, wo.z));
    const float g2 = wo.z * length(float3(alpha * wi.xy, wi.z));
    return (2.0f * wo.z) / (g1 + g2);
}

// -----------------------------------------------------------------
// NDF

// Returns D(h)
float ggx_NDF(float ndoth_squared, float alpha_squared) {
    return alpha_squared /
           (MERIAN_PI * pow2((ndoth_squared * alpha_squared) + (1.0f - ndoth_squared)));
}

// Returns D(h) for per-axis alpha; division-free form from Heitz 2018,
// https://jcgt.org/published/0007/04/01/ (Sampling the GGX Distribution of Visible Normals).
float ggx_NDF(const float3 h, const float2 alpha) {
    const float alpha_xy = alpha.x * alpha.y;
    const float3 v = float3(alpha.y * h.x, alpha.x * h.y, alpha_xy * h.z);
    const float w2 = alpha_xy / dot(v, v);
    return alpha_xy * w2 * w2 / MERIAN_PI;
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

/**
 * The VNDF pdf for an outgoing direction (has the Jacobian of the reflection applied), for
 * per-axis alpha. Takes the half vector, not wo. Assumes a local space with n = (0, 0, 1).
 */
float ggx_VNDF_pdf_wo(const float3 wi, const float3 h, const float2 alpha) {
    return (ggx_NDF(h, alpha) * smith_g1_over_widotn(wi, alpha)) / 4.0f;
}

// -----------------------------------------------------------------

// Approximate pre-integrated GGX BRDF (Ray Tracing Gems Ch. 32, Eq. 4). Returns F0 · scale + bias.
// Adapted from:
// https://github.com/NVIDIAGameWorks/Falcor/blob/eb540f6748774680ce0039aaf3ac9279266ec521/Source/Falcor/Rendering/Materials/IsotropicGGX.slang#L215
float3 ggx_integral_approx_(const float3 F0,
                            const float alpha,
                            const float cosTheta,
                            const float2x2 M1,
                            const float3x3 M2,
                            const float2x2 M3,
                            const float3x3 M4) {
    const float c = abs(cosTheta);

    const float4 X = float4(1.0f, c, c * c, c * c * c);
    const float4 Y = float4(1.0f, alpha, alpha * alpha, alpha * alpha * alpha);

    float bias = dot(mul(M1, X.xy), Y.xy) / dot(mul(M2, X.xyw), Y.xyw);
    const float scale = dot(mul(M3, X.xy), Y.xy) / dot(mul(M4, X.xzw), Y.xyw);

    // Hack to suppress bias when F0 is ~0 (no specular reflectance).
    const float F0_luma = (F0.r + F0.g + F0.b) * (1.0f / 3.0f);
    bias *= saturate(F0_luma * 50.0f);

    return F0 * max(0.0f, scale) + max(0.0f, bias);
}

// Height-correlated Smith G2 variant. Use this when paired with smith_g2_over_widotn.
float3 ggx_integral_approx_correlated(const float3 F0, const float alpha, const float cosTheta) {
    const float2x2 M1 = float2x2(0.995367f, -1.38839f, -0.24751f, 1.97442f);
    const float3x3 M2 = float3x3(1.0f, 2.68132f, 52.366f, 16.0932f, -3.98452f, 59.3013f, -5.18731f,
                                 255.259f, 2544.07f);
    const float2x2 M3 = float2x2(-0.0564526f, 3.82901f, 16.91f, -11.0303f);
    const float3x3 M4 = float3x3(1.0f, 4.11118f, -1.37886f, 19.3254f, -28.9947f, 16.9514f,
                                 0.545386f, 96.0994f, -79.4492f);
    return ggx_integral_approx_(F0, alpha, cosTheta, M1, M2, M3, M4);
}

// Separable Smith G2 = G1(wi)·G1(wo) variant.
float3 ggx_integral_approx_separable(const float3 F0, const float alpha, const float cosTheta) {
    const float2x2 M1 = float2x2(0.99044f, -1.28514f, 1.29678f, -0.755907f);
    const float3x3 M2 = float3x3(1.0f, 2.92338f, 59.4188f, 20.3225f, -27.0302f, 222.592f, 121.563f,
                                 626.13f, 316.627f);
    const float2x2 M3 = float2x2(0.0365463f, 3.32707f, 9.0632f, -9.04756f);
    const float3x3 M4 = float3x3(1.0f, 3.59685f, -1.36772f, 9.04401f, -16.3174f, 9.22949f, 5.56589f,
                                 19.7886f, -20.2123f);
    return ggx_integral_approx_(F0, alpha, cosTheta, M1, M2, M3, M4);
}

// A specular microfacet BRDF with GGX normal distribution. alpha is per-axis (tangent,
// bitangent); a scalar broadcast yields the isotropic lobe at no extra cost.
struct GGXBRDF : BSDF {
    float2 alpha;

    __init(float alpha) {
        this.alpha = float2(alpha);
    }

    __init(float2 alpha) {
        this.alpha = alpha;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return float3(0.0f);
        }

        const float3 h = normalize(wo + wi);
        const float G2_over_widotn = smith_g2_over_widotn(wo, wi, alpha);
        const float D = ggx_NDF(h, alpha);

        // wo.z cancels
        return float3((D * G2_over_widotn) / 4.0f);
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

    float pdf(const float3 wi, const float3 wo) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return 0.0f;
        }

        const float3 h = normalize(wi + wo);
        return ggx_VNDF_pdf_wo(wi, h, alpha);
    }

    override Optional<ShadingFunctionSample> sample_eval(const float3 wi,
                                                         inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return none;
        }

        const float3 h = ggx_VNDF_sample_H(wi, alpha, rng.next_float2());
        const float3 wo = reflect2(wi, h, dot(wi, h));
        if (wo.z < BSDF_MIN_COS_THETA) {
            return none;
        }

        // the NDF and cosines cancel, so it stays finite even at alpha -> 0.
        const float pdf = ggx_VNDF_pdf_wo(wi, h, alpha);
        const float weight = smith_g2_over_widotn(wo, wi, alpha) / smith_g1_over_widotn(wi, alpha);
        return ShadingFunctionSample(wo, pdf, float3(weight), BSDFLobe::Glossy);
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(get_preintegrated_specular(float3(1.0f), wi));
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        // the fit is isotropic; sqrt(alpha_x * alpha_y) preserves the NDF peak.
        return ggx_integral_approx_correlated(F0, sqrt(alpha.x * alpha.y), wi.z);
    }

    property float roughness {
        get { return sqrt(min(alpha.x, alpha.y)); }
    }
}

// A specular microfacet BTDF with GGX normal distribution.
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
struct GGXBTDF : BSDF {
    float alpha;
    float eta;
    float3 tint;

    __init(float alpha, float eta, float3 tint) {
        this.alpha = alpha;
        this.eta = eta;
        this.tint = tint;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA || wo.z > -BSDF_MIN_COS_THETA) {
            return float3(0.0f);
        }

        float3 h = normalize(wo + (wi * eta));
        h *= sign(h.z);
        const float widoth = dot(wi, h);
        const float wodoth = dot(wo, h);
        const float alpha_squared = pow2(alpha);
        const float G2_over_widotn = smith_g2_over_widotn(abs(wo.z), wi.z, alpha_squared);
        const float D = ggx_NDF(pow2(h.z), alpha_squared);

        const float denom = wodoth + (eta * widoth);
        const float jacobian = pow2(eta) * widoth * wodoth / pow2(denom);
        // wo.z cancels
        return tint * (D * G2_over_widotn * abs(jacobian));
    }

    Optional<float3> sample(const float3 wi, inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return none;
        }

        const float3 h = ggx_VNDF_sample_H(wi, alpha, rng.next_float2());
        const float3 wo = refract(-wi, h, eta);
        if (all(wo == float3(0.0f)) || wo.z > -BSDF_MIN_COS_THETA) {
            return none; // total internal reflection
        }

        return wo;
    }

    float pdf(const float3 wi, const float3 wo) {
        if (wi.z < BSDF_MIN_COS_THETA || wo.z > -BSDF_MIN_COS_THETA) {
            return 0.0f;
        }

        float3 h = normalize(wo + (wi * eta));
        h *= sign(h.z);
        const float wodoth = dot(wo, h);
        const float denom = wodoth + (eta * dot(wi, h));
        /* * abs(wodoth) / pow2(denom) is the Jacobian of the refraction */
        return ggx_VNDF_H_pdf(wi, h, pow2(alpha)) * abs(wodoth) / pow2(denom);
    }

    override Optional<ShadingFunctionSample> sample_eval(const float3 wi,
                                                         inout RandomGenerator rng) {
        if (wi.z < BSDF_MIN_COS_THETA) {
            return none;
        }

        const float alpha_squared = pow2(alpha);
        const float3 h = ggx_VNDF_sample_H(wi, alpha, rng.next_float2());
        const float3 wo = refract(-wi, h, eta);
        if (all(wo == float3(0.0f)) || wo.z > -BSDF_MIN_COS_THETA) {
            return none;
        }

        const float wodoth = dot(wo, h);
        const float denom = wodoth + (eta * dot(wi, h));
        const float pdf = ggx_VNDF_H_pdf(wi, h, alpha_squared) * abs(wodoth) / pow2(denom);
        // Like GGXBRDF, the VNDF weight f(wi,wo) * |wo.z| / pdf reduces to G2(wi,wo) / G1(wi); the
        // refraction adds an eta^2 radiance factor. It stays finite even at alpha -> 0.
        const float weight = pow2(eta) * smith_g2_over_widotn(abs(wo.z), wi.z, alpha_squared) /
                             smith_g1_over_widotn(wi.z, alpha_squared);
        return ShadingFunctionSample(wo, pdf, tint * weight, BSDFLobe::Glossy);
    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(pow2(eta) * tint);
    }

    float3 get_preintegrated_specular(const float3 F0, const float3 wi) {
        return float3(0.0f);
    }

    property float roughness {
        get { return sqrt(alpha); }
    }
}

typealias SpecularBRDF = GGXBRDF;
typealias SpecularBTDF = GGXBTDF;

} // namespace merian
