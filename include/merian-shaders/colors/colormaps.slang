namespace merian {

// Polynomial fit of the Turbo colormap (Mikhailov, Google AI Blog 2019).
float3 colormap_turbo(float x) {
    x = saturate(x);
    const float4 v4 = float4(1.f, x, x * x, x * x * x);
    const float2 v2 = v4.zw * v4.z;
    return float3(dot(v4, float4(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f)) +
                      dot(v2, float2(-152.94239396f, 59.28637943f)),
                  dot(v4, float4(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f)) +
                      dot(v2, float2(4.27729857f, 2.82956604f)),
                  dot(v4, float4(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f)) +
                      dot(v2, float2(-89.90310912f, 27.34824973f)));
}

// Polynomial fit of matplotlib's plasma (Zucker 2020, "Colormap polynomials", shadertoy WlfXRN).
float3 colormap_plasma(float x) {
    x = saturate(x);
    const float3 c0 = float3(0.05873234392399702f, 0.02333670892565664f, 0.5433401826748754f);
    const float3 c1 = float3(2.176514634195958f, 0.2383834171260182f, 0.7539604599784036f);
    const float3 c2 = float3(-2.689460476458034f, -7.455851135738909f, 3.110799939717086f);
    const float3 c3 = float3(6.130348345893603f, 42.3461881477227f, -28.51885465332158f);
    const float3 c4 = float3(-11.10743619062271f, -82.66631109428045f, 60.13984767418263f);
    const float3 c5 = float3(10.02306557647065f, 71.41361770095349f, -54.07218655560067f);
    const float3 c6 = float3(-3.658713842777788f, -22.93153465461149f, 18.19190778539828f);
    return saturate(c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6))))));
}

// Diverging blue-white-red map for signed quantities; 0.5 = neutral.
float3 colormap_diverging(float x) {
    x = saturate(x);
    const float3 lo = float3(0.10f, 0.25f, 0.85f);
    const float3 mid = float3(0.95f, 0.95f, 0.95f);
    const float3 hi = float3(0.85f, 0.15f, 0.10f);
    return x < 0.5f ? lerp(lo, mid, x * 2.f) : lerp(mid, hi, (x - 0.5f) * 2.f);
}

} // namespace merian
