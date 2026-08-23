#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.scene.scene;
import merian_shaders.utils.pseudorandom;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;
import merian_shaders.colors.colorspaces;

using merian;

struct LightUpdateBinding {
    RWStructuredBuffer<EmissiveTriangle, ScalarDataLayout> triangles;
    StructuredBuffer<LightGeometry, ScalarDataLayout> light_geometries;
    uint light_geometry_count;
    uint triangle_count;
    // emission evaluations per triangle for the flux estimate
    uint flux_samples;
};

// light geometry containing triangle index i (upper_bound over first_triangle)
uint find_light_geometry(const ParameterBlock<LightUpdateBinding> params, const uint i) {
    uint lo = 0;
    uint hi = params.light_geometry_count - 1;
    while (lo < hi) {
        const uint mid = (lo + hi + 1) >> 1;
        if (params.light_geometries[mid].first_triangle <= i)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

// Rebuilds every emissive triangle record (world-space positions, uvs, flux) from the scene.
[shader("compute")][numthreads(64, 1, 1)] void main(uint3 thread_id : SV_DispatchThreadID,
                                                    ParameterBlock<Scene> scene,
                                                    ParameterBlock<LightUpdateBinding> params) {
    const uint i = thread_id.x;
    if (i >= params.triangle_count)
        return;

    const LightGeometry lg = params.light_geometries[find_light_geometry(params, i)];
    const uint primitive_id = i - lg.first_triangle;
    const GeometryData gd = scene.get_geometry_data(lg.geometry_id);

    const uint3 indices = scene.get_vertex_indices(gd, primitive_id);
    const PackedVertexData v0 = scene.get_packed_vertex_data(gd, indices.x);
    const PackedVertexData v1 = scene.get_packed_vertex_data(gd, indices.y);
    const PackedVertexData v2 = scene.get_packed_vertex_data(gd, indices.z);

    EmissiveTriangle tri;
    tri.p0 = scene.to_world_position(gd, lg.instance_index, v0.position);
    tri.uv0 = v0.uv;
    // wind so that cross(p1 - p0, p2 - p0) faces front
    if (gd.flags & GeometryDataFlags::FlipFacing) {
        tri.p1 = scene.to_world_position(gd, lg.instance_index, v2.position);
        tri.p2 = scene.to_world_position(gd, lg.instance_index, v1.position);
        tri.uv1 = v2.uv;
        tri.uv2 = v1.uv;
    } else {
        tri.p1 = scene.to_world_position(gd, lg.instance_index, v1.position);
        tri.p2 = scene.to_world_position(gd, lg.instance_index, v2.position);
        tri.uv1 = v1.uv;
        tri.uv2 = v2.uv;
    }
    tri.geometry_id = lg.geometry_id;
    tri.primitive_id = primitive_id;
    tri.material_id = gd.material_id;
    const bool two_sided = (gd.flags & GeometryDataFlags::TwoSided) != 0;
    tri.flags = two_sided ? EmissiveTriangleFlags::EmissiveTwoSided : EmissiveTriangleFlags(0);

    // flux = pi * area * mean radiance (diffuse emitter) per visible face, estimated at
    // stratified points. Cutout texels are no surface (the tracer passes through them) and
    // contribute nothing.
    const float3 scaled_n = tri.get_scaled_normal();
    const float area2 = length(scaled_n);
    float flux = 0.f;
    if (area2 > 0.f) {
        const float3 n = scaled_n / area2;
        RandomGenerator rng = PCG(xxhash32(uint2(lg.geometry_id, primitive_id)));
        const float2 shift = rng.next_float2();
        float luminance_sum = 0.f;
        for (uint s = 0; s < params.flux_samples; s++) {
            // R2 sequence, randomly shifted per triangle, warped to uniform barycentrics
            const float2 u = frac(r2_sequence(s) + shift);
            const float su = sqrt(u.x);
            const float3 bary = float3(1.f - su, su * (1.f - u.y), su * u.y);
            const float3 pos = tri.get_position(bary);
            const float2 uv = tri.get_uv(bary);
            if (scene.material_system.alpha_test(gd.material_id, uv))
                continue;
            luminance_sum += yuv_luminance(scene.material_system.get_emission(
                gd.material_id,
                ShadingData(n, pos, uv, scene.time, OrthonormalFrame(n), n, uint16_t(1), true),
                DefaultLODSampler()));
            if (two_sided) {
                luminance_sum += yuv_luminance(scene.material_system.get_emission(
                    gd.material_id,
                    ShadingData(-n, pos, uv, scene.time, OrthonormalFrame(-n), -n, uint16_t(1),
                                false),
                    DefaultLODSampler()));
            }
        }
        flux = MERIAN_PI * 0.5f * area2 * luminance_sum / float(params.flux_samples);
    }
    tri.flux = isfinite(flux) ? flux : 0.f;

    params.triangles[i] = tri;
}
