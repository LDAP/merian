#include "merian-shaders/utils/math-constants.slangh"

namespace merian {

// Equal-area octahedral mapping [Clarberg 2008]; every texel covers 4 pi / N^2 sr
float2 equal_area_sphere_to_square(const float3 d) {
    const float3 a = abs(d);
    const float r = sqrt(max(0.f, 1.f - a.z));
    const float m = max(a.x, a.y);
    const float b = m == 0.f ? 0.f : min(a.x, a.y) / m;
    const float phi_b = atan(b) * (2.f / MERIAN_PI);
    const float phi = a.x < a.y ? 1.f - phi_b : phi_b;
    const float v = phi * r;
    const float u = r - v;
    const float2 folded = d.z < 0.f ? float2(1.f - v, 1.f - u) : float2(u, v);
    return 0.5f * float2((d.x < 0.f ? -folded.x : folded.x) + 1.f,
                         (d.y < 0.f ? -folded.y : folded.y) + 1.f);
}

float3 equal_area_square_to_sphere(const float2 p) {
    const float u = 2.f * p.x - 1.f;
    const float v = 2.f * p.y - 1.f;
    const float up = abs(u);
    const float vp = abs(v);
    const float signed_distance = 1.f - (up + vp);
    const float r = 1.f - abs(signed_distance);
    const float phi = (r == 0.f ? 1.f : (vp - up) / r + 1.f) * (MERIAN_PI / 4.f);
    const float z = (1.f - r * r) * sign(signed_distance);
    const float s = r * sqrt(max(0.f, 2.f - r * r));
    return float3(cos(phi) * s * sign(u), sin(phi) * s * sign(v), z);
}

// Axis aligned cubemap indices
// 0: pos x, 1: neg x, 2: pos y, 3: neg: y, 4: pos z, 5: neg z
uint cubemap_side(const float3 w) {
    const float3 a = abs(w);
    const bool xd = a.x > a.y && a.x > a.z, yd = a.y > a.z;
    return (xd ? 0u : yd ? 2u : 4u) + uint((xd ? w.x : yd ? w.y : w.z) < 0.0);
}

// 45 deg turned cubemap indices
uint cubemap_side_45deg(const float3 w) {
    const float s = w.x + w.y;
    const float t = MERIAN_M_SQRT2 * w.z;
    const float3 d = float3(MERIAN_M_SQRT2 * (w.x - w.y), s - t, s + t);
    const float3 a = abs(d);
    const bool xd = a.x > a.y && a.x > a.z, yd = a.y > a.z;
    return (xd ? 0u : yd ? 2u : 4u) + uint((xd ? d.x : yd ? d.y : d.z) < 0.0);
}

} // namespace merian
