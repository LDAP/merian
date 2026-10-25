import merian_shaders.utils.encoding;
import merian_shaders.utils.frame;

namespace merian {

// Direction <-> 2D map transforms with the pole at +z, for directional histograms and
// environment-style images. All uv in [0,1)^2; w = 0 marks points outside the mapped domain.

static const uint DIR_MAP_OCT = 0;     // full sphere, octahedral
static const uint DIR_MAP_LATLONG = 1; // full sphere, equirectangular
static const uint DIR_MAP_HEMI = 2;    // +z hemisphere, Lambert azimuthal equal-area

// Map-space frame: rows transform world -> map space (z = pole).
// local = shading frame constructed from the normal, world = y-up rotated onto the pole.
float3x3 dir_map_frame(const float3 normal, const bool local_frame) {
    if (local_frame) {
        const OrthonormalFrame frame = OrthonormalFrame(normal);
        return float3x3(frame.get_tangent(), frame.get_bitangent(), frame.get_normal());
    }
    return float3x3(float3(1, 0, 0), float3(0, 0, -1), float3(0, 1, 0));
}

float4 dir_map_dir(const uint transform, const float2 uv) {
    switch (transform) {
    case DIR_MAP_LATLONG: {
        const float phi = (uv.x * 2.f - 1.f) * 3.14159265f;
        const float theta = uv.y * 3.14159265f;
        return float4(sin(theta) * sin(phi), sin(theta) * cos(phi), cos(theta), 1.f);
    }
    case DIR_MAP_HEMI: {
        const float2 p = uv * 2.f - 1.f;
        const float r2 = dot(p, p);
        if (r2 > 1.f) {
            return float4(0, 0, 1, 0);
        }
        // z = 1 - r^2 is the equal-area profile of the Lambert azimuthal projection,
        // sin(theta) = r * sqrt(2 - r^2)
        return float4(p * sqrt(max(2.f - r2, 0.f)), 1.f - r2, 1.f);
    }
    default: {
        const float2 p = uv * 2.f - 1.f;
        float3 v = float3(p, 1.f - abs(p.x) - abs(p.y));
        if (v.z < 0.f) {
            v.xy = copysign(1.f - abs(v.yx), v.xy);
        }
        return float4(normalize(v), 1.f);
    }
    }
}

float3 dir_map_uv(const uint transform, const float3 dir) {
    switch (transform) {
    case DIR_MAP_LATLONG: {
        const float phi = atan2(dir.x, dir.y);
        const float theta = acos(clamp(dir.z, -1.f, 1.f));
        return float3(phi / (2.f * 3.14159265f) + 0.5f, theta / 3.14159265f, 1.f);
    }
    case DIR_MAP_HEMI: {
        if (dir.z < 0.f) {
            return float3(0, 0, 0);
        }
        const float r = sqrt(max(1.f - dir.z, 0.f));
        const float xy_len = length(dir.xy);
        const float2 p = xy_len > 1e-6f ? dir.xy / xy_len * r : float2(0);
        return float3(p * 0.5f + 0.5f, 1.f);
    }
    default:
        return float3(octahedral_project(dir) * 0.5f + 0.5f, 1.f);
    }
}

// Solid angle of the spherical triangle abc (Van Oosterom and Strackee 1983).
float dir_map_triangle_solid_angle(const float3 a, const float3 b, const float3 c) {
    const float triple = abs(dot(a, cross(b - a, c - a)));
    return 2.f * atan2(triple, 1.f + dot(a, b) + dot(b, c) + dot(c, a));
}

float dir_map_unit_circle(const float x) {
    return sqrt(max((1.f - x) * (1.f + x), 0.f));
}

// Integral of sqrt(1 - t^2) over [a, b] for 0 <= a <= b <= 1, without cancellation for b - a small.
float dir_map_unit_circle_integral(const float a, const float b) {
    const float ga = dir_map_unit_circle(a);
    const float gb = dir_map_unit_circle(b);
    const float q = ga + gb > 0.f ? a * (a + b) / (ga + gb) : 0.f;
    // b g(b) - a g(a) and asin(b) - asin(a) = asin(b g(a) - a g(b))
    return 0.5f * ((b - a) * (gb - q) + asin(min((b - a) * (ga + q), 1.f)));
}

// Area of the unit disk in a bin its circle crosses, the bin within one quadrant.
float dir_map_disk_bin_area(const float2 lo, const float2 hi) {
    // folded to x <= y, where the circle's slope stays below 1
    float2 inner = min(abs(lo), abs(hi));
    float2 outer = max(abs(lo), abs(hi));
    if (inner.x + outer.x > inner.y + outer.y) {
        inner = inner.yx;
        outer = outer.yx;
    }
    const float full_until = clamp(dir_map_unit_circle(outer.y), inner.x, outer.x);
    const float empty_from = clamp(dir_map_unit_circle(inner.y), inner.x, outer.x);
    return (outer.y - inner.y) * (full_until - inner.x) +
           dir_map_unit_circle_integral(full_until, empty_from) -
           inner.y * (empty_from - full_until);
}

// Solid angle covered by one bin, exact for an even res.
float dir_map_bin_solid_angle(const uint transform, const uint2 bin, const uint res) {
    const float inv_res = 1.f / float(res);
    switch (transform) {
    case DIR_MAP_LATLONG: {
        // azimuth step times cos(theta_0) - cos(theta_1)
        const float theta_step = 3.14159265f * inv_res;
        const float theta_mid = (float(bin.y) + 0.5f) * theta_step;
        return 2.f * 3.14159265f * inv_res * 2.f * sin(theta_mid) * sin(0.5f * theta_step);
    }
    case DIR_MAP_HEMI: {
        // equal-area: 2 sr per unit of disk area
        const float2 lo = float2(bin) * (2.f * inv_res) - 1.f;
        const float2 hi = lo + 2.f * inv_res;
        // a bin belongs to the disk if one of its quarter centres does
        const float2 nearest = min(abs(0.75f * lo + 0.25f * hi), abs(0.25f * lo + 0.75f * hi));
        const float2 farthest = max(abs(lo), abs(hi));
        if (dot(nearest, nearest) >= 1.f) {
            return 0.f;
        }
        if (dot(farthest, farthest) <= 1.f) {
            return 8.f * inv_res * inv_res;
        }
        return 2.f * max(dir_map_disk_bin_area(lo, hi), 0.f);
    }
    default: {
        // the edges and the diagonal along the octahedron's folds are great circles
        const float3 c00 = dir_map_dir(DIR_MAP_OCT, float2(bin) * inv_res).xyz;
        const float3 c10 = dir_map_dir(DIR_MAP_OCT, float2(bin + uint2(1, 0)) * inv_res).xyz;
        const float3 c11 = dir_map_dir(DIR_MAP_OCT, float2(bin + uint2(1, 1)) * inv_res).xyz;
        const float3 c01 = dir_map_dir(DIR_MAP_OCT, float2(bin + uint2(0, 1)) * inv_res).xyz;
        if ((2 * bin.x + 1 < res) == (2 * bin.y + 1 < res)) {
            return dir_map_triangle_solid_angle(c00, c10, c01) +
                   dir_map_triangle_solid_angle(c10, c11, c01);
        }
        return dir_map_triangle_solid_angle(c00, c10, c11) +
               dir_map_triangle_solid_angle(c00, c11, c01);
    }
    }
}

} // namespace merian
