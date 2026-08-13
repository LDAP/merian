import merian_shaders.utils.encoding;
import merian_shaders.utils.frame;

namespace merian {

// Direction <-> 2D map transforms with the pole at +z, for directional histograms and
// environment-style images. All uv in [0,1)^2; w = 0 marks points outside the mapped domain.

static const uint DIR_MAP_OCT = 0;     // full sphere, octahedral
static const uint DIR_MAP_LATLONG = 1; // full sphere, equirectangular
static const uint DIR_MAP_HEMI = 2;    // +z hemisphere, Lambert azimuthal equal-area

// Map-space frame: rows transform world -> map space (z = pole).
// local = shading frame constructed from the normal, world = y-up swizzled to the pole.
float3x3 dir_map_frame(const float3 normal, const bool local_frame) {
    if (local_frame) {
        const OrthonormalFrame frame = OrthonormalFrame(normal);
        return float3x3(frame.get_tangent(), frame.get_bitangent(), frame.get_normal());
    }
    return float3x3(float3(1, 0, 0), float3(0, 0, 1), float3(0, 1, 0));
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

// Solid angle covered by one bin: finite-difference Jacobians at 2x2 sub-bin points, so bins
// straddling the domain boundary (hemisphere rim) get their partial coverage. A sub-point whose
// stencil crosses the boundary is skipped entirely: differencing a real direction against the
// out-of-domain sentinel would explode the Jacobian.
float dir_map_bin_solid_angle(const uint transform, const uint2 bin, const uint res) {
    const float inv_res = 1.f / float(res);
    const float h = 0.1f * inv_res;
    float sum = 0.f;
    [unroll]
    for (uint qy = 0; qy < 2; qy++) {
        [unroll]
        for (uint qx = 0; qx < 2; qx++) {
            const float2 uv = (float2(bin) + float2(0.25f + 0.5f * qx, 0.25f + 0.5f * qy)) *
                              inv_res;
            const float4 xp = dir_map_dir(transform, uv + float2(h, 0));
            const float4 xn = dir_map_dir(transform, uv - float2(h, 0));
            const float4 yp = dir_map_dir(transform, uv + float2(0, h));
            const float4 yn = dir_map_dir(transform, uv - float2(0, h));
            if (dir_map_dir(transform, uv).w == 0.f || xp.w == 0.f || xn.w == 0.f ||
                yp.w == 0.f || yn.w == 0.f) {
                continue;
            }
            sum += length(cross(xp.xyz - xn.xyz, yp.xyz - yn.xyz)) / (4.f * h * h);
        }
    }
    return sum * 0.25f * inv_res * inv_res;
}

} // namespace merian
