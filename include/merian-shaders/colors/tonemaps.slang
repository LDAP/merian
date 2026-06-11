import "merian-shaders/colors/colorspaces.slang";
import "merian-shaders/utils/math.slang";

namespace merian {

float3 tonemap_clamp(float3 rgb) {
    return saturate(rgb);
}

float3 tonemap_reinhard(float3 rgb, float white) {
    float l = yuv_luminance(rgb);
    float l_new = (l * (1.0f + (l / (white * white)))) / (1.0f + l);
    return rgb * l_new * safe_reciprocal(l);
}

float3 _tonemap_uncharted2(const float3 x) {
    const float A = 0.15;
    const float B = 0.50;
    const float C = 0.10;
    const float D = 0.20;
    const float E = 0.02;
    const float F = 0.30;

    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// Hable Tone Mapping / Hable Filmic
float3 tonemap_uncharted2(float3 rgb, float exposure_bias, float W) {
    return _tonemap_uncharted2(rgb * exposure_bias) / _tonemap_uncharted2(W);
}

// Adapted from Stephen Hill (@self_shadow)
float3 tonemap_aces(float3 rgb) {
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    float3 col =
        mul(rgb, float3x3(float3(0.59719, 0.07600, 0.02840), float3(0.35458, 0.90834, 0.13383),
                          float3(0.04823, 0.01566, 0.83777)));

    const float3 a = col * (col + 0.0245786f) - 0.000090537f;
    const float3 b = col * (0.983729f * col + 0.4329510f) + 0.238081f;
    col = a / b;

    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    return saturate(
        mul(col, float3x3(float3(1.60475, -0.10208, -0.00327), float3(-0.53108, 1.10813, -0.07276),
                          float3(-0.07367, -0.00605, 1.07602))));
}

/*
Default parameters:
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
*/
float3 tonemap_aces_approx(float3 rgb, float a, float b, float c, float d, float e) {
    const float3 col = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e);
    return saturate(col);
}

// AgX (Troy Sobotka) minimal implementation, Benjamin Wrensch
// https://iolite-engine.com/blog_posts/minimal_agx_implementation
float3 _agx_default_contrast_approx(float3 x) {
    const float3 x2 = x * x;
    const float3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 +
           0.1191 * x - 0.00232;
}

// ASC CDL look: only slope is per-channel (Golden preset tints it), offset/power/sat are scalar.
float3 tonemap_agx(float3 rgb, float3 slope, float offset, float power, float sat) {
    const float3x3 agx_mat =
        float3x3(float3(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
                 float3(0.0784335999999992, 0.878468636469772, 0.0784336),
                 float3(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
    const float3x3 agx_mat_inv =
        float3x3(float3(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
                 float3(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
                 float3(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));

    const float min_ev = -12.47393;
    const float max_ev = 4.026069;

    // 1. inset and encode into log2 space
    float3 val = mul(rgb, agx_mat);
    val = clamp(log2(val), min_ev, max_ev);
    val = (val - min_ev) / (max_ev - min_ev);

    // 2. sigmoid contrast curve
    val = _agx_default_contrast_approx(val);

    // 3. ASC CDL look + saturation in the encoded space (Rec.709 luma)
    const float luma = dot(val, float3(0.2126, 0.7152, 0.0722));
    val = pow(max(val * slope + offset, 0.0), power);
    val = luma + sat * (val - luma);

    // 4. outset and linearize (EOTF), output is linear for the display OETF
    val = mul(val, agx_mat_inv);
    return pow(max(val, 0.0), 2.2);
}

// Lottes 2016, "Advanced Techniques and Optimization of HDR Color Pipelines" (AMD)
// https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/d7531ae47d8b36a5d4025663e731a47a38be882f/framework/cauldron/framework/inc/shaders/tonemapping/tonemappers.hlsl#L21
float3 tonemap_lottes(
    in float3 rgb, float contrast, float shoulder, float hdrMax, float midIn, float midOut) {
    const float b =
        -((-pow(midIn, contrast) +
           (midOut * (pow(hdrMax, contrast * shoulder) * pow(midIn, contrast) -
                      pow(hdrMax, contrast) * pow(midIn, contrast * shoulder) * midOut)) /
               (pow(hdrMax, contrast * shoulder) * midOut -
                pow(midIn, contrast * shoulder) * midOut)) /
          (pow(midIn, contrast * shoulder) * midOut));
    const float c =
        (pow(hdrMax, contrast * shoulder) * pow(midIn, contrast) -
         pow(hdrMax, contrast) * pow(midIn, contrast * shoulder) * midOut) /
        (pow(hdrMax, contrast * shoulder) * midOut - pow(midIn, contrast * shoulder) * midOut);

    rgb = min(rgb, float3(hdrMax)); // fix for shoulder > 1
    float peak = max(rgb.r, max(rgb.g, rgb.b));

    peak = max(1e-7f, peak);

    float3 ratio = rgb / peak;
    const float z = pow(peak, contrast);
    peak = z / (pow(z, shoulder) * b + c);

    const float crosstalk = 4.0;       // controls amount of channel crosstalk
    const float saturation = contrast; // full tonal range saturation control
    const float crossSaturation = contrast * 16.0;

    float white = 1.0;

    ratio = pow(abs(ratio), float3(saturation / crossSaturation));
    ratio = lerp(ratio, float3(white), pow(peak, crosstalk));
    ratio = pow(abs(ratio), float3(crossSaturation));

    return peak * ratio;
}

}
