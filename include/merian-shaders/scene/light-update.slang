#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.scene.scene;
import merian_shaders.utils.pseudorandom;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;
import merian_shaders.colors.colorspaces;

using merian;

struct LightUpdateBinding {
    RWStructuredBuffer<EmissiveTriangle, ScalarDataLayout> triangles;
    RWStructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    RWStructuredBuffer<EmissiveRegion, ScalarDataLayout> regions;
    StructuredBuffer<LightGeometry, ScalarDataLayout> light_geometries;
    uint light_geometry_count;
    uint triangle_count;
    // emission evaluations per triangle for the flux estimate
    uint flux_samples;
    uint lanes;
};

// Rebuilds every emissive triangle record (world-space positions, uvs, flux) from the scene.
[shader("compute")][numthreads(LIGHT_UPDATE_GROUP, 1, 1)] void
main(uint3 group_id : SV_GroupID,
     uint thread_id : SV_GroupIndex,
     ParameterBlock<Scene> scene,
     ParameterBlock<LightUpdateBinding> params) {
    const uint lanes = params.lanes;
    const uint lane = thread_id & (lanes - 1u);
    const uint i = (group_id.x + group_id.y * LIGHT_DISPATCH_ROW) * (LIGHT_UPDATE_GROUP / lanes) +
                   thread_id / lanes;

    uint geometry = merian::find_light_geometry(params.light_geometries,
                                                params.light_geometry_count, WaveReadLaneFirst(i));
    if (i >= params.triangle_count)
        return;
    while (geometry + 1u < params.light_geometry_count &&
           params.light_geometries[geometry + 1u].first_triangle <= i)
        geometry++;
    const LightGeometry lg = params.light_geometries[geometry];
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
    const bool alpha_tested = (gd.flags & GeometryDataFlags::AlphaTested) != 0;
    tri.flags = EmissiveTriangleFlags(
        (two_sided ? uint16_t(EmissiveTriangleFlags::EmissiveTwoSided) : uint16_t(0)) |
        (alpha_tested ? uint16_t(EmissiveTriangleFlags::EmissiveAlphaTested) : uint16_t(0)));

    // flux = pi * area * mean radiance (diffuse emitter) per visible face, estimated at
    // stratified points. Cutout texels are no surface (the tracer passes through them) and
    // contribute nothing.
    const float3 scaled_n = tri.get_scaled_normal();
    const float area2 = length(scaled_n);
    float luminance_sum = 0.f;
    EmissiveRegion region = EmissiveRegion::whole();
    region.swapped = uint16_t((gd.flags & GeometryDataFlags::FlipFacing) != 0 ? 1 : 0);
    if (lg.emission_records != nullptr) {
        const EmissionAddress record = lg.emission_records + lg.emission_offsets[primitive_id];
        const uint count = record[EMISSION_RECORD_COUNT];
        if (count < 1u << (2 * lg.emission_level)) {
            region.record = record;
            region.level = uint16_t(lg.emission_level);
            region.count = count;
        }
    }
    if (area2 > 0.f && region.get_fraction() > 0.f) {
        const float3 n = scaled_n / area2;
        const OrthonormalFrame front = OrthonormalFrame(n);
        const OrthonormalFrame back = OrthonormalFrame(-n);
        let material = scene.material_system.get_material(gd.material_id);
        RandomGenerator rng = PCG(xxhash32(uint2(lg.geometry_id, primitive_id)));
        const float3 shift = rng.next_float3();
        for (uint s = lane; s < params.flux_samples; s += lanes) {
            // R2 sequence, randomly shifted per triangle, warped to uniform barycentrics
            const float2 u = frac(r2_sequence(s) + shift.xy);
            const float su = sqrt(u.x);
            const float3 bary = region.to_triangle(
                float3(1.f - su, su * (1.f - u.y), su * u.y),
                frac((float(s) + shift.z) / float(params.flux_samples)));
            const float3 pos = tri.get_position(bary);
            const float2 uv = tri.get_uv(bary);
            if (alpha_tested && scene.material_system.alpha_test(gd.material_id, uv))
                continue;
            luminance_sum += yuv_luminance(material.get_emission(
                ShadingData(n, pos, uv, scene.time, front, n, uint16_t(1), true),
                scene.material_system.texture_manager, DefaultLODSampler()));
            if (two_sided) {
                luminance_sum += yuv_luminance(material.get_emission(
                    ShadingData(-n, pos, uv, scene.time, back, -n, uint16_t(1), false),
                    scene.material_system.texture_manager, DefaultLODSampler()));
            }
        }
    }

    float total = luminance_sum;
    for (uint offset = lanes / 2; offset > 0; offset >>= 1)
        total += WaveShuffle(total, WaveGetLaneIndex() ^ offset);
    if (lane != 0)
        return;
    const float flux =
        area2 > 0.f && params.flux_samples > 0
            ? MERIAN_PI * 0.5f * area2 * region.get_fraction() * total / float(params.flux_samples)
            : 0.f;
    tri.flux = isfinite(flux) ? flux : 0.f;

    params.triangles[i] = tri;
    params.regions[i] = region;

    EmissiveLightProxy proxy;
    proxy.centroid = (tri.p0 + tri.p1 + tri.p2) * (1.f / 3.f);
    proxy.flux = tri.flux;
    params.proxies[i] = proxy;
}
