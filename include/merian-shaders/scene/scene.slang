#pragma once

__exported import merian_shaders.scene.camera;
__exported import merian_shaders.scene.acceleration_structure;
__exported import merian_shaders.scene.environment_map;
__exported import merian_shaders.shading.materials.material_system;
__exported import merian_shaders.scene.scene_data;

import merian_shaders.utils.encoding;

namespace merian {

struct Hit {
    uint instance_id;
    uint instance_index;
    uint primitive_id;
    uint geometry_index;
    float2 barycentrics;

    __init(uint instance_id,
           uint instance_index,
           uint primitive_id,
           uint geometry_index,
           float2 barycentrics) {
        this.instance_id = instance_id;
        this.instance_index = instance_index;
        this.primitive_id = primitive_id;
        this.geometry_index = geometry_index;
        this.barycentrics = barycentrics;
    }

    // invalid hit
    __init() {
        this.instance_id = uint(-1);
        this.instance_index = uint(-1);
        this.primitive_id = uint(-1);
        this.geometry_index = uint(-1);
        this.barycentrics = float2(0);
    }

    bool is_valid() {
        return instance_id != uint(-1);
    }

    // Returns (1 - u - v, u, v) for vertex interpolation
    float3 get_barycentrics3() {
        return float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    }

    GeometryID get_geometry_id() {
        return instance_id + geometry_index;
    }
}

// TODO: use link-time type once Slang's lookupExternDeclRefType is fixed
// extern struct SceneAccelerationStructure : AccelerationStructure;
typedef HWAccelerationStructure SceneAccelerationStructure;

// // See scene.hpp for details.
struct Scene {
    SceneAccelerationStructure as;

    StructuredBuffer<GeometryData, ScalarDataLayout> geometries;

    // One transform per TLAS instance, indexed by InstanceIndex().
    StructuredBuffer<float4x4, ScalarDataLayout> instance_transforms;
    StructuredBuffer<float4x4, ScalarDataLayout> inverse_transposed_instance_transforms;
    StructuredBuffer<float4x4, ScalarDataLayout> prev_instance_transforms;
    StructuredBuffer<float4x4, ScalarDataLayout> prev_inverse_transposed_instance_transforms;

    ParameterBlock<MaterialSystem> material_system;
    EnvMap env_map;

    Camera camera;
    Camera prev_camera;

    // Global scene parameters
    uint frame;
    float time;
    float time_diff;

    // Global heterogeneous fog parameters
    float3 mu_s;
    float mu_t;

    // -----------------------------------------

    /* Traces the ray agains the scenes acceleration structure.
     *
     * Returns the hit information.
     */
    Hit trace_ray(const RayDesc ray,
                  const uint ray_flags = RAY_FLAG_NONE,
                  const uint instances_mask = 0xff) {
        SceneAccelerationStructure::RayQueryImpl rq;
        as.trace_ray(rq, ray, ray_flags, instances_mask);
        _proceed_ray(rq);

        if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            return Hit(rq.CommittedInstanceID(), rq.CommittedInstanceIndex(),
                       rq.CommittedPrimitiveIndex(), rq.CommittedGeometryIndex(),
                       rq.CommittedRayBarycentrics());
        }

        return Hit();
    }

    /* Traces the ray agains the scenes acceleration structure.
     *
     * Returns if visible, i.e. if nothing was hit.
     */
    bool trace_visibility_ray(const RayDesc ray,
                              const uint ray_flags = RAY_FLAG_NONE,
                              const uint instances_mask = 0xff) {
        SceneAccelerationStructure::RayQueryImpl rq;
        as.trace_ray(rq, ray, ray_flags, instances_mask);
        _proceed_ray(rq);

        return rq.CommittedStatus() == COMMITTED_NOTHING;
    }

    // -----------------------------------------

    // Loads the geometry data from memory.
    GeometryData get_geometry_data(const GeometryID geometry_id) {
        return geometries[geometry_id];
    }

    // Loads the geometry data from memory.
    GeometryData get_geometry_data(const uint instance_id, const uint geometry_index) {
        return get_geometry_data(instance_id + geometry_index);
    }

    // Loads the instance transform for a TLAS instance from memory.
    float4x4 get_instance_transform(const uint instance_index) {
        return instance_transforms[instance_index];
    }

    // Loads the inverse transposed instance transform for a TLAS instance from memory (for
    // transforming directions / normals).
    float4x4 get_inverse_transposed_instance_transform(const uint instance_index) {
        return inverse_transposed_instance_transforms[instance_index];
    }

    // Loads the previous frame instance transform for a TLAS instance from memory.
    float4x4 get_prev_instance_transform(const uint instance_index) {
        return prev_instance_transforms[instance_index];
    }

    // Loads the previous frame inverse transposed instance transform for a TLAS instance.
    float4x4 get_prev_inverse_transposed_instance_transform(const uint instance_index) {
        return prev_inverse_transposed_instance_transforms[instance_index];
    }

    // Conventience method for get_geometry_data(...).material_id;.
    MaterialID get_material_id(const GeometryID geometry_id) {
        return get_geometry_data(geometry_id).material_id;
    }

    // Convenience method for get_geometry_data(...).material_id;
    MaterialID get_material_id(const Hit hit) {
        return get_material_id(hit.get_geometry_id());
    }

    // Loads the vertex indices from memory via buffer device address.
    uint3 get_vertex_indices(const GeometryData geometry_data, const uint primitive_id) {
        return geometry_data.indices[primitive_id];
    }

    // Loads the vertex data from memory via buffer device address.
    PackedVertexData get_packed_vertex_data(const GeometryData geometry_data,
                                            const uint vertex_index) {
        return geometry_data.vertices[vertex_index];
    }

    vector<PackedVertexData, 3> get_packed_vertex_data(const GeometryData geometry_data,
                                                       const uint primitive_id,
                                                       const float3 barycentrics) {
        const uint3 indices = get_vertex_indices(geometry_data, primitive_id);
        return { get_packed_vertex_data(geometry_data, indices.x),
                 get_packed_vertex_data(geometry_data, indices.y),
                 get_packed_vertex_data(geometry_data, indices.z) };
    }

    VertexData get_interpolated_vertex_data(const GeometryData geometry_data,
                                            const uint primitive_id,
                                            const float3 barycentrics) {
        const uint3 indices = get_vertex_indices(geometry_data, primitive_id);

        VertexData v = barycentrics.x * get_packed_vertex_data(geometry_data, indices.x).unpack();
        v += barycentrics.y * get_packed_vertex_data(geometry_data, indices.y).unpack();
        v += barycentrics.z * get_packed_vertex_data(geometry_data, indices.z).unpack();

        v.normal = normalize(v.normal);
        v.tangent = normalize(v.tangent);

        return v;
    }

    // Convenience method to load the vertex uvs.
    float2 get_interpolated_uv(const GeometryData geometry_data,
                               const uint primitive_id,
                               const float3 barycentrics) {
        const uint3 indices = get_vertex_indices(geometry_data, primitive_id);

        return barycentrics.x * float2(geometry_data.vertices[indices.x].uv) +
               barycentrics.y * float2(geometry_data.vertices[indices.y].uv) +
               barycentrics.z * float2(geometry_data.vertices[indices.z].uv);
    }

    // Previous-frame object-space position for a vertex.
    // For static meshes (prev_vertices == null) this equals the current position.
    float3 get_prev_object_position(const GeometryData geometry_data, const uint vertex_index) {
        if (uint64_t(geometry_data.prev_vertices) == 0) {
            return geometry_data.vertices[vertex_index].position;
        }
        return geometry_data.prev_vertices[vertex_index].position;
    }

    float3 get_interpolated_prev_object_position(const GeometryData geometry_data,
                                                 const uint primitive_id,
                                                 const float3 barycentrics) {
        const uint3 indices = get_vertex_indices(geometry_data, primitive_id);
        return barycentrics.x * get_prev_object_position(geometry_data, indices.x) +
               barycentrics.y * get_prev_object_position(geometry_data, indices.y) +
               barycentrics.z * get_prev_object_position(geometry_data, indices.z);
    }

    // -----------------------------------------

    // Transform object-space position to world space.
    float3
    to_world_position(const GeometryData gd, const uint instance_index, const float3 object_pos) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_pos;
        return mul(get_instance_transform(instance_index), float4(object_pos, 1.0)).xyz;
    }

    // Transform object-space direction to world space.
    float3
    to_world_direction(const GeometryData gd, const uint instance_index, const float3 object_dir) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_dir;
        return normalize(
            mul(get_inverse_transposed_instance_transform(instance_index), float4(object_dir, 0.0))
                .xyz);
    }

    // Transform object-space position to previous frame world space.
    float3 to_prev_world_position(const GeometryData gd,
                                  const uint instance_index,
                                  const float3 object_pos) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_pos;
        return mul(get_prev_instance_transform(instance_index), float4(object_pos, 1.0)).xyz;
    }

    // Transform object-space direction to previous frame world space.
    float3 to_prev_world_direction(const GeometryData gd,
                                   const uint instance_index,
                                   const float3 object_dir) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_dir;
        return normalize(mul(get_prev_inverse_transposed_instance_transform(instance_index),
                             float4(object_dir, 0.0))
                             .xyz);
    }

    // -----------------------------------------

    // NOTE: Using concrete RayQuery instead of generic <RQ : IRayQuery> to work around
    // Slang SPIR-V codegen crash in IRBuilder::getSet() with generics (Slang 2026.5.2).
    void _proceed_ray<uint FLAGS>(inout RayQuery<FLAGS> rq) {
        while (rq.Proceed()) {

            if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                const GeometryData geometry_data =
                    get_geometry_data(rq.CandidateInstanceID(), rq.CandidateGeometryIndex());

                float2 bary = rq.CandidateRayBarycentrics();
                float3 bary3 = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

                if (material_system.alpha_test(
                        geometry_data.material_id,
                        get_interpolated_uv(geometry_data, rq.CandidatePrimitiveIndex(), bary3))) {
                    continue;
                }

                rq.CommitNonOpaqueTriangleHit();
            }
        }
    }
}

}

