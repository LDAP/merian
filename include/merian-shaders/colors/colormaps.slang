namespace merian {

// Polynomial fit of the Turbo colormap (Mikhailov, Google AI Blog 2019); the host mirror
// with the same coefficients is imgui_colormaps.cpp.
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

// Diverging blue-white-red map for signed quantities; 0.5 = neutral.
float3 colormap_diverging(float x) {
    x = saturate(x);
    const float3 lo = float3(0.10f, 0.25f, 0.85f);
    const float3 mid = float3(0.95f, 0.95f, 0.95f);
    const float3 hi = float3(0.85f, 0.15f, 0.10f);
    return x < 0.5f ? lerp(lo, mid, x * 2.f) : lerp(mid, hi, (x - 0.5f) * 2.f);
}

} // namespace merian
