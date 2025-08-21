#pragma once

implementing colorspaces;

namespace merian {

/* The following is darktable Uniform Color Space 2022
 * © Aurélien Pierre
 * https://eng.aurelienpierre.com/2022/02/color-saturation-control-for-the-21th-century/
 *
 * Use this space for color-grading in a perceptual framework.
 * The CAM terms have been removed for performance.
 *
 * From darktable, licensed under MIT licence.
 */

public float Y_to_dt_UCS_L_star(float Y) {
    // WARNING: L_star needs to be < 2.098883786377, meaning Y needs to be < 3.875766378407574e+19
    float Y_hat = pow(Y, 0.631651341915130615234375f);
    return (2.0988838672637939453125f * Y_hat) / (Y_hat + 1.12426769733428955078125f);
}

public float dt_UCS_L_star_to_Y(float L_star) {
    // WARNING: L_star needs to be < 2.098883786377, meaning Y needs to be < 3.875766378407574e+19
    return pow((1.12426769733428955078125f * L_star) / (2.0988838672637939453125f - L_star),
               1.58315181732177734375f);
}

public float2 xy_to_dt_UCS_UV(float2 xy) {
    float3 uvd = mul(float3(xy, 1.0f),
                     float3x3(float3(-0.783941002840055, 0.745273540913283, 0.318707282433486),
                              float3(0.277512987809202, -0.205375866083878, 2.16743692732158),
                              float3(0.153836578598858, -0.165478376301988, 0.291320554395942)));
    uvd.xy /= uvd.z;

    const float2 factors = float2(1.39656225667, 1.4513954287);
    const float2 half_values = float2(1.49217352929, 1.52488637914);
    float2 UV_star = factors * uvd.xy / (abs(uvd.xy) + half_values);

    return mul(UV_star, float2x2(float2(-1.12498390674591064453125f, 1.86323320865631103515625f),
                                 float2(-0.980483710765838623046875f, 1.97185313701629638671875f)));
}

//  input :
//    * xyY in normalized CIE XYZ for the 2° 1931 observer adapted for D65
//    * L_white the lightness of white as dt UCS L* lightness
//    * cz = 1 for standard pre-print proofing conditions with average surround and n = 20 %
//            (background = middle grey, white = perfect diffuse white)
//  range : xy in [0; 1], Y normalized for perfect diffuse white = 1
public float3 xyY_to_dt_UCS_JCH(float3 xyY, float L_white) {
    float2 param = xyY.xy;
    float2 UV_star_prime = xy_to_dt_UCS_UV(param);
    float L_star = Y_to_dt_UCS_L_star(xyY.z);
    float M2 = dot(UV_star_prime, UV_star_prime);
    return float3(L_star / L_white,
                  ((15.93299388885498046875f * pow(L_star, 0.6523997783660888671875f)) *
                   pow(M2, 0.6007556915283203125f)) /
                      L_white,
                  atan2(UV_star_prime.y, UV_star_prime.x));
}

//  input :
//    * xyY in normalized CIE XYZ for the 2° 1931 observer adapted for D65
//    * L_white the lightness of white as dt UCS L* lightness
//    * cz = 1 for standard pre-print proofing conditions with average surround and n = 20 %
//            (background = middle grey, white = perfect diffuse white)
//  range : xy in [0; 1], Y normalized for perfect diffuse white = 1
public float3 dt_UCS_JCH_to_xyY(float3 JCH, float L_white) {
    float L_star = JCH.x * L_white;
    float M =
        pow((JCH.y * L_white) / (15.93299388885498046875f * pow(L_star, 0.6523997783660888671875f)),
            0.83228504657745361328125f);
    float2 UV_star = float2(cos(JCH.z), sin(JCH.z)) * M;
    UV_star = mul(UV_star, float2x2(float2(-5.03752231597900390625f, 4.760029315948486328125f),
                                    float2(-2.5048563480377197265625f, 2.87401294708251953125f)));
    float2 UV = (float2(-1.49217355251312255859375f, -1.5248863697052001953125f) * UV_star) /
                (abs(UV_star) - float2(1.39656221866607666015625f, 1.45139539241790771484375f));
    float3 xyD =
        mul(float3(UV, 1.0f),
            float3x3(float3(0.167171472114775, -0.150959086409163, 0.940254742367256),
                     float3(0.141299802443708, -0.155185060382272, 1.000000000000000),
                     float3(-0.00801531300850582, -0.00843312433578007, -0.0256325967652889)));
    return float3(xyD.xy / xyD.z.xx, dt_UCS_L_star_to_Y(L_star));
}

public float3 dt_UCS_JCH_to_HSB(float3 JCH) {
    float3 HSB;
    HSB.z = JCH.x * (pow(JCH.y, 1.33654224872589111328125f) + 1.0f);
    float _218;
    if (HSB.z > 0.0f) {
        _218 = JCH.y / HSB.z;
    } else {
        _218 = 0.0f;
    }
    return float3(JCH.z, _218, HSB.z);
}

public float3 dt_UCS_HSB_to_JCH(float3 HSB) {
    return float3(HSB.z / (pow(HSB.y * HSB.z, 1.33654224872589111328125f) + 1.0f), HSB.y * HSB.z,
                  HSB.x);
}

public float3 dt_UCS_JCH_to_HCB(float3 JCH) {
    return float3(JCH.zy, JCH.x * (pow(JCH.y, 1.33654224872589111328125f) + 1.0f));
}

public float3 dt_UCS_HCB_to_JCH(float3 HCB) {
    return float3(HCB.z / (pow(HCB.y, 1.33654224872589111328125f) + 1.0f), HCB.yx);
}

}
