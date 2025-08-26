[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;

struct PushConstant {
    float brightness;
    float contrast;
    float bias;
    int colourmode;
};

[vk::push_constant]
ConstantBuffer<PushConstant> params;

[vk::binding(0)]
Sampler2D<float4> img_in;
[vk::binding(1)]
RWTexture2D<float4> img_out;

#ifdef WRITE_CURVE
[vk::binding(2)]
RWTexture2D<float4> img_crv;
#endif

import merian_shaders.utils.textures;
import merian_shaders.colors.colorspaces;

using merian;

float weibull_cdf(float x,  // input value (0, infty)
                  float il, // weibull 1.0/lambda, scale parameter (0, infty)
                  float k)  // weibull k, shape parameter          (0, infty)
{
    return 1.0 - exp(-pow(max(x, 1e-7) * il, k));
}

float // derivative of cdf:
weibull_pdf(float x, float il, float k) {
    x = max(x, 1e-7);
    return k * il * pow(x * il, k - 1.0) * exp(-pow(x * il, k));
}
float3 // vector version
weibull_cdf(float3 x, float il, float k) {
    return 1.0 - exp(-pow(max(x, 1e-7) * il, float3(k)));
}

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(const int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, img_out))
        return;

    float3 col0 = img_in.Load(int3(ipos, 0)).rgb;
    // Operates in rec2020
    col0 = rec709_to_rec2020(col0);

    float il = max(5e-3, params.brightness);
    float k = max(1e-5, params.contrast);

    col0 += params.bias;
    float3 col1 = weibull_cdf(col0, il, k);
    if (params.colourmode == 0) { // colour using aurelien's patented ucs:
        const float3x3 xyz_to_rec2020 = float3x3(1.7166511880, -0.6666843518, 0.0176398574,

                                                 -0.3556707838, 1.6164812366, -0.0427706133,

                                                 -0.2533662814, 0.0157685458, 0.9421031212);
        const float3x3 rec2020_to_xyz = float3x3(0.6369580483, 0.262700212, 0,

                                                 0.1446169036, 0.6779980715, 0.0280726931,

                                                 0.1688809752, 0.0593017165, 1.0609850578);
        float3 xyz0 = mul(col0, rec2020_to_xyz);
        float3 xyz1 = mul(col1, rec2020_to_xyz);
        float3 xyY0 = float3(xyz0.xy / max(1e-4, dot(float3(1), xyz0)), xyz0.y);
        float3 xyY1 = float3(xyz1.xy / max(1e-4, dot(float3(1), xyz1)), xyz0.y);
        const float L_white = 1.0;
        float3 jch0 = xyY_to_dt_UCS_JCH(xyY0, L_white);
        float3 jch1 = xyY_to_dt_UCS_JCH(xyY1, L_white);
        jch1 = float3(jch1.x, jch1.y, jch0.z);
        xyY1 = dt_UCS_JCH_to_xyY(jch1, L_white);
        xyz1 = float3(xyY1.xy, 1.0 - xyY1.x - xyY1.y) * xyz1.y / max(1e-4, xyY1.y);
        col1 = mul(xyz1, xyz_to_rec2020);
    } else if (params.colourmode == 2) { // colour reconstruction using munsell hue constancy.
        float3 xyY0 = rec2020_to_xyY(col0);
        float2 m0 = xy_to_munsell(xyY0.xy);
        float3 xyY1 = rec2020_to_xyY(col1);
        float2 m1 = xy_to_munsell(xyY1.xy);
        xyY1.xy = munsell_to_xy(float2(m0.x, m1.y));
        col1 = xyY_to_rec2020(xyY1);
    } else if (params.colourmode == 3) { // simple rgb/hsv hack:
        col1 = adjust_colour_dng(col0, col1);
    } else if (params.colourmode == 1) { // apply per channel
    }

#ifdef WRITE_CURVE
    if (all(lessThan(ipos, imageSize(img_crv)))) {
        float4 col_crv = float4(0, 0, 0, 1);
        float of0 = 0; // 0.1; // extend range outside [0,1]?
        float off = of0 * imageSize(img_crv).x;
        float x = ipos.x / float(imageSize(img_crv).x - off);
        float y = weibull_cdf(x, il, k);
        float ddx = weibull_pdf(x, il, k);
        const float w = 2.0;                 // stroke width
        float t = w * sqrt(1.0 + ddx * ddx); // line thinckness along const y
        float d = abs((1.0 + of0 - y) * (1.0 - of0) * imageSize(img_crv).y - ipos.y);
        col_crv = max(col_crv, 0.2 * float4(float3(smoothstep(t, 0.0, d)), 1));
        imageStore(img_crv, ipos, col_crv);
    }
#endif

    col1 = rec2020_to_rec709(col1);
    img_out.Store(ipos, float4(col1, 1));
}

