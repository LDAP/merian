#pragma once

#include "merian-shaders/utils/math-constants.slangh"

import "merian-shaders/bsdfs/bsdf.slang";
import "merian-shaders/utils/math.slang";

namespace merian {

float bsdf_ggx_roughness_to_alpha(float roughness) {
    return pow2(roughness);
}

// Smiths shadow masking term
float ggx_smith_g1(float minuswidotn, float alpha_squared) {
    return (2.0f * minuswidotn) /
           (minuswidotn +
            sqrt(alpha_squared + ((1.0f - alpha_squared) * (minuswidotn * minuswidotn))));
}

// Smiths shadow masking term
float ggx_smith_g1_over_minuswidotn(float minuswidotn, float alpha_squared) {
    return 2.0f / (minuswidotn +
                   sqrt(alpha_squared + ((1.0f - alpha_squared) * (minuswidotn * minuswidotn))));
}

// minuswi = V
float ggx_smith_g2_over_minuswidotn(float wodotn, float minuswidotn, float alpha_squared) {
    float g1 = minuswidotn * sqrt(alpha_squared + ((1.0f - alpha_squared) * (wodotn * wodotn)));
    float g2 =
        wodotn * sqrt(alpha_squared + ((1.0f - alpha_squared) * (minuswidotn * minuswidotn)));
    return (2.0f * wodotn) / (g1 + g2);
}

float ggx_D(float ndoth_squared, float alpha_squared) {
    return alpha_squared /
           (MERIAN_PI * pow2((ndoth_squared * alpha_squared) + (1.0f - ndoth_squared)));
}

// -----------------------------------------------------------------

float bsdf_ggx_times_wodotn(float wodotn, float minuswidotn, float ndoth, float alpha) {
    float G2 = ggx_smith_g2_over_minuswidotn(wodotn, minuswidotn, alpha * alpha);
    float D = ggx_D(ndoth * ndoth, alpha * alpha);
    return (D * G2) / 4.0f;
}

float bsdf_ggx_times_wodotn(float3 wi, float3 wo, float3 n, float alpha) {
    float3 h = normalize(wo - wi);
    float wodotn = dot(wo, n);
    float minuswidotn = dot(-wi, n);
    float ndoth = dot(n, h);
    return bsdf_ggx_times_wodotn(wodotn, minuswidotn, ndoth, alpha);
}

float bsdf_ggx_times_wodotn(float3 wi, float3 wo, float3 n, float alpha, float F0) {
    float3 h = normalize(wo - wi);
    float wodotn = dot(wo, n);
    float minuswidotn = dot(-wi, n);
    float ndoth = dot(n, h);
    float bsdf = bsdf_ggx_times_wodotn(wodotn, minuswidotn, ndoth, alpha);
    float minuswidoth = dot(-wi, h);
    float F = F0 + ((1.0f - F0) * pow(max(1.0f - minuswidoth, 0.0f), 5.0f));
    return F * bsdf;
}

float3 bsdf_ggx_times_wodotn(float3 wi, float3 wo, float3 n, float alpha, float3 F0) {
    float3 h = normalize(wo - wi);
    float wodotn = dot(wo, n);
    float minuswidotn = dot(-wi, n);
    float ndoth = dot(n, h);
    float bsdf = bsdf_ggx_times_wodotn(wodotn, minuswidotn, ndoth, alpha);
    float minuswidoth = dot(-wi, h);
    float3 F = F0 + ((1.0f.xxx - F0) * pow(max(1.0f - minuswidoth, 0.0f), 5.0f));
    return F * bsdf;
}


struct GGXBRDF : BSDF {
    float3 albedo;

    __init(float3 albedo) {
        this.albedo = albedo;
    }

    float3 eval(const float3 wi, const float3 wo, inout RandomGenerator rng) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return float3(0.0);
        }

    }

    Optional<BSDFSample> sample(const float3 wi, inout RandomGenerator rng) {


        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return none;
        }

        return BSDFSample(wo, );
    }

    float pdf(const float3 wi, const float3 wo) {
        if (min(wi.z, wo.z) < BSDF_MIN_COS_THETA) {
            return 0.0;
        }

    }

    Albedo get_albedo(const float3 wi) {
        return Albedo(albedo);
    }
}


float bsdf_ggx_diffuse_mix_times_wodotn(float3 wi, float3 wo, float3 n, float alpha, float F0) {
    float3 h = normalize(wo - wi);
    float wodotn = clamp(dot(n, wo), 0.0f, 1.0f);
    if (wodotn <= 0.0f) {
        return 0.0f;
    }
    float minuswidotn = dot(-wi, n);
    float ndoth = dot(n, h);
    float minuswidoth = dot(-wi, h);
    float F = F0 + ((1.0f - F0) * pow(max(1.0f - minuswidoth, 0.0f), 5.0f));
    return lerp(0.3183098733425140380859375f * wodotn,
                bsdf_ggx_times_wodotn(wodotn, minuswidotn, ndoth, alpha), F);
}

float3 bsdf_ggx_VNDF_H_sample(float3 minuswi, float2 alpha, float2 random) {
    float3 wiStd = normalize(float3(minuswi.xy * alpha, minuswi.z));
    float phi = ((2.0f * random.x) - 1.0f) * MERIAN_PI;
    float z = mad(1.0f - random.y, 1.0f + wiStd.z, -wiStd.z);
    float sinTheta = sqrt(clamp(1.0f - (z * z), 0.0f, 1.0f));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float3 c = float3(x, y, z);
    float3 wmStd = c + wiStd;
    float3 wm = normalize(float3(wmStd.xy * alpha, wmStd.z));
    return wm;
}

float3 bsdf_ggx_VNDF_H_sample(float3 minuswi, float3 n, float alpha, float2 random) {
    float3 wi_z = n * dot(minuswi, n);
    float3 wi_xy = minuswi - wi_z;
    float3 wiStd = normalize(wi_z - (wi_xy * alpha));
    float wiStd_z = dot(wiStd, n);
    float phi = ((2.0f * random.x) - 1.0f) * MERIAN_PI;
    float z = ((1.0f - random.y) * (1.0f + wiStd_z)) - wiStd_z;
    float sinTheta = sqrt(clamp(1.0f - (z * z), 0.0f, 1.0f));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float3 cStd = float3(x, y, z);
    float3 wr = n + float3(0.0f, 0.0f, 1.00000011920928955078125f);
    float3 c = ((wr * dot(wr, cStd)) / wr.z.xxx) - cStd;
    float3 wmStd = c + wiStd;
    float3 wmStd_z = n * dot(n, wmStd);
    float3 wmStd_xy = wmStd_z - wmStd;
    float3 wm = normalize(wmStd_z + (wmStd_xy * alpha));
    return wm;
}

float3 bsdf_ggx_VNDF_sample(float3 wi, float3 n, float alpha, float2 random) {
    float3 h = bsdf_ggx_VNDF_H_sample(-wi, n, alpha, random);
    return reflect(wi, h);
}

float bsdf_ggx_VNDF_pdf(float minuswidotn, float ndoth_squared, float alpha_squared) {
    float G1 = bsdf_ggx_smith_g1_over_minuswidotn(minuswidotn, alpha_squared);
    float D = bsdf_ggx_D(ndoth_squared, alpha_squared);
    return (D * G1) / 4.0f;
}

float bsdf_ggx_VNDF_pdf(float3 wi, float3 wo, float3 n, float alpha) {
    float3 h = normalize(wo - wi);
    float minuswidotn = dot(-wi, n);
    float ndoth = clamp(dot(n, h), 0.0f, 1.0f);
    return bsdf_ggx_VNDF_pdf(minuswidotn, ndoth * ndoth, alpha * alpha);
}

float bsdf_ggx_diffuse_mix_pdf(float3 wi, float3 wo, float3 n, float alpha) {
    float wodotn = dot(wo, n);
    if (wodotn <= 0.0f) {
        return 0.0f;
    }
    float3 h = normalize(wo - wi);
    float minuswidotn = dot(-wi, n);
    float ndoth = clamp(dot(n, h), 0.0f, 1.0f);
    float diffuse_pdf = 0.3183098733425140380859375f * wodotn;
    float ggx_vndf_pdf = bsdf_ggx_VNDF_pdf(minuswidotn, ndoth * ndoth, alpha * alpha);
    float fresnel = 0.0199999995529651641845703125f +
                    (0.980000019073486328125f * pow(max(1.0f - minuswidotn, 0.0f), 5.0f));
    return lerp(diffuse_pdf, ggx_vndf_pdf, fresnel);
}

float3 bsdf_ggx_diffuse_mix_sample(float3 wi, float3 n, float alpha, float3 random) {
    float fresnel = 0.0199999995529651641845703125f +
                    (0.980000019073486328125f * pow(max(1.0f - dot(-wi, n), 0.0f), 5.0f));
    if (fresnel < random.x) {
        return mul(sample_cos(random.yz), make_frame(n));
    } else {
        return bsdf_ggx_VNDF_sample(wi, n, alpha, random.yz);
    }
}

}
