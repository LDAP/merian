__exported import merian_shaders.scene.camera;
__exported import merian_shaders.scene.acceleration_structure;
__exported import merian_shaders.scene.environment_map;
__exported import merian_shaders.shading.materials.material_system;
__exported import merian_shaders.scene.scene_data;
__exported import merian_shaders.scene.hit;

import merian_shaders.utils.encoding;

// Workaround for https://github.com/shader-slang/slang/pull/10769; scene.cpp injects replacements.
__exported import scene_as_workaround;
__exported import scene_env_map_workaround;

namespace merian {

// Once https://github.com/shader-slang/slang/pull/10769 lands these can replace the workaround imports above.
// extern struct SceneAccelerationStructure : AccelerationStructure = NullAccelerationStructure;
// extern struct SceneEnvMap : IEnvMap = EmptyEnv;

// See scene.hpp for details.
struct Scene {
    SceneAccelerationStructure as;

    StructuredBuffer<GeometryData, ScalarDataLayout> geometries;

    // One transform per TLAS instance, indexed by InstanceIndex().
    StructuredBuffer<float4x4, ScalarDataLayout> instance_transforms;
    StructuredBuffer<float4x4, ScalarDataLayout> inverse_transposed_instance_transforms;
    StructuredBuffer<float4x4, ScalarDataLayout> prev_instance_transforms;
    StructuredBuffer<float4x4, ScalarDataLayout> prev_inverse_transposed_instance_transforms;

    ParameterBlock<MaterialSystem> material_system;
    SceneEnvMap env_map;

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

    // Loads the vertex indices from memory via buffer device address, or
    // computes them from primitive_id for IndexType::None (no memory access).
    uint3 get_vertex_indices(const GeometryData geometry_data, const uint primitive_id) {
        if (geometry_data.flags & GeometryDataFlags::IndexTypeUint32) {
            uint3* p = (uint3*)geometry_data.indices;
            return p[primitive_id];
        }
        if (geometry_data.flags & GeometryDataFlags::IndexTypeUint16) {
            vector<uint16_t, 3>* p = (vector<uint16_t, 3>*)geometry_data.indices;
            return uint3(p[primitive_id]);
        }
        if (geometry_data.flags & GeometryDataFlags::IndexTypeUint8) {
            vector<uint8_t, 3>* p = (vector<uint8_t, 3>*)geometry_data.indices;
            return uint3(p[primitive_id]);
        }
        const uint base = primitive_id * 3;
        return uint3(base, base + 1, base + 2);
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
        v.tangent.xyz = normalize(v.tangent.xyz);

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
    // For non-morphed meshes this equals the current position.
    float3 get_prev_object_position(const GeometryData geometry_data, const uint vertex_index) {
        if ((geometry_data.flags & GeometryDataFlags::HasPrevVertices) == 0)
            return geometry_data.vertices[vertex_index].position;
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

    // Transform an object-space normal to world space (covariant: uses inverse-transpose).
    float3
    to_world_normal(const GeometryData gd, const uint instance_index, const float3 object_normal) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_normal;
        return normalize(mul(get_inverse_transposed_instance_transform(instance_index),
                             float4(object_normal, 0.0))
                             .xyz);
    }

    // Transform an object-space vector (edge, tangent, ...) to world space (uses regular M).
    float3
    to_world_vector(const GeometryData gd, const uint instance_index, const float3 object_vec) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_vec;
        return mul((float3x3)get_instance_transform(instance_index), object_vec);
    }

    // Transform object-space position to previous frame world space.
    float3 to_prev_world_position(const GeometryData gd,
                                  const uint instance_index,
                                  const float3 object_pos) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_pos;
        if ((gd.flags & GeometryDataFlags::IsAnimated) == 0)
            return mul(get_instance_transform(instance_index), float4(object_pos, 1.0)).xyz;
        return mul(get_prev_instance_transform(instance_index), float4(object_pos, 1.0)).xyz;
    }

    // Transform an object-space normal to previous-frame world space.
    float3 to_prev_world_normal(const GeometryData gd,
                                const uint instance_index,
                                const float3 object_normal) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_normal;
        if ((gd.flags & GeometryDataFlags::IsAnimated) == 0)
            return normalize(mul(get_inverse_transposed_instance_transform(instance_index),
                                 float4(object_normal, 0.0))
                                 .xyz);
        return normalize(mul(get_prev_inverse_transposed_instance_transform(instance_index),
                             float4(object_normal, 0.0))
                             .xyz);
    }

    // Transform an object-space vector (edge, tangent, ...) to previous-frame world space.
    float3 to_prev_world_vector(const GeometryData gd,
                                const uint instance_index,
                                const float3 object_vec) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_vec;
        if ((gd.flags & GeometryDataFlags::IsAnimated) == 0)
            return mul((float3x3)get_instance_transform(instance_index), object_vec);
        return mul((float3x3)get_prev_instance_transform(instance_index), object_vec);
    }

    // -----------------------------------------

    // none on miss
    Optional<GeometryData> get_geometry_data(const Hit hit) {
        if (!hit.is_valid()) return none;
        return get_geometry_data(hit.instance_id, hit.geometry_index);
    }

    // Hit must be a valid triangle hit.
    TriangleHit get_triangle_hit(const Hit hit) {
        return get_triangle_hit(hit, get_geometry_data(hit.instance_id, hit.geometry_index));
    }

    TriangleHit get_triangle_hit(const Hit hit, const GeometryData gd) {
        const uint3 indices = get_vertex_indices(gd, hit.primitive_id);
        TriangleHit tri;
        tri.gd = gd;
        tri.v0 = get_packed_vertex_data(gd, indices.x);
        tri.v1 = get_packed_vertex_data(gd, indices.y);
        tri.v2 = get_packed_vertex_data(gd, indices.z);
        return tri;
    }

    Surface get_surface(const Hit hit, const TriangleHit tri) {
        const float3 bary = hit.get_barycentrics3();

        const float3 pos = to_world_position(tri.gd, hit.instance_index, tri.get_position(hit));
        const float3 edge01 = to_world_vector(tri.gd, hit.instance_index, tri.get_triangle_edge01());
        const float3 edge02 = to_world_vector(tri.gd, hit.instance_index, tri.get_triangle_edge02());

        float3 face_normal = normalize(cross(edge01, edge02));
        if (tri.gd.flags & GeometryDataFlags::FlipFacing) face_normal = -face_normal;

        float3 vertex_normal;
        if (tri.gd.flags & GeometryDataFlags::FlatShading) {
            vertex_normal = face_normal;
        } else {
            const float3 n_obj = normalize(
                bary.x * decode_normal(tri.v0.encoded_normal) +
                bary.y * decode_normal(tri.v1.encoded_normal) +
                bary.z * decode_normal(tri.v2.encoded_normal));
            vertex_normal = to_world_normal(tri.gd, hit.instance_index, n_obj);
        }

        // per-vertex tangent or Lengyel UV-derived fallback
        float3 tangent;
        float handedness;
        bool tangent_valid;
        if (tri.gd.flags & GeometryDataFlags::HasTangents) {
            const float4 t0 = decode_tangent(tri.v0.encoded_tangent);
            const float4 t1 = decode_tangent(tri.v1.encoded_tangent);
            const float4 t2 = decode_tangent(tri.v2.encoded_tangent);
            const float4 t = bary.x * t0 + bary.y * t1 + bary.z * t2;
            tangent = to_world_vector(tri.gd, hit.instance_index, normalize(t.xyz));
            handedness = sign(t.w);
            tangent_valid = true;
        } else {
            const float2 duv1 = float2(tri.v1.uv) - float2(tri.v0.uv);
            const float2 duv2 = float2(tri.v2.uv) - float2(tri.v0.uv);
            const float det = duv1.x * duv2.y - duv2.x * duv1.y;
            if (abs(det) > 1e-8f) {
                const float winding_sign =
                    (tri.gd.flags & GeometryDataFlags::FlipFacing) ? 1.f : -1.f;
                tangent = (duv2.y * edge01 - duv1.y * edge02) / det;
                handedness = sign(det) * winding_sign;
                tangent_valid = true;
            } else {
                tangent = float3(1, 0, 0);
                handedness = 1.f;
                tangent_valid = false;
            }
        }

        Surface s;
        s.material_id = tri.gd.material_id;
        s.pos = pos;
        s.uv = tri.get_uv(hit);
        s.face_normal = face_normal;
        s.frame = tangent_valid ? OrthonormalFrame(vertex_normal, tangent, handedness)
                                : OrthonormalFrame(vertex_normal);
        return s;
    }

    float3 get_prev_world_position(const Hit hit, const GeometryData gd) {
        const float3 prev_obj =
            get_interpolated_prev_object_position(gd, hit.primitive_id, hit.get_barycentrics3());
        return to_prev_world_position(gd, hit.instance_index, prev_obj);
    }

    // Pixel-space delta from prev_camera projection to current pixel, jitter-corrected.
    float2 compute_motion_vector(const float3 prev_pos, const uint2 pixel, const uint2 dim) {
        return prev_camera.project_pinhole(prev_pos, dim) - float2(pixel)
             + prev_camera.jitter - camera.jitter;
    }

    ShadingPoint get_shading_point(const RayDesc ray,
                                   const Hit hit,
                                   const GeometryData gd,
                                   const uint16_t bounce = 0,
                                   const LODSampler2D lod = DefaultLODSampler()) {
        const TriangleHit tri = get_triangle_hit(hit, gd);
        const Surface surface = get_surface(hit, tri);

        // const float3 prev_pos = get_prev_world_position(hit, gd);

        const ShadingData sd = ShadingData(-ray.Direction,
                                           surface.pos,
                                           surface.uv,
                                           time,
                                           surface.frame,
                                           surface.face_normal,
                                           bounce);

        ShadingPoint sp;
        sp.pos = surface.pos;
        sp.enc_face_normal = encode_normal(surface.face_normal);
        // sp.prev_pos = prev_pos;
        sp.material = material_system.get_material_sample(surface.material_id, sd, lod);
        return sp;
    }

    ShadingPoint get_shading_point(const RayDesc ray,
                                   const Hit hit,
                                   const uint16_t bounce = 0,
                                   const LODSampler2D lod = DefaultLODSampler()) {
        if (let gd = get_geometry_data(hit)) {
            if ((gd.flags & GeometryDataFlags::UseEnvMap) == 0) {
                return get_shading_point(ray, hit, gd, bounce, lod);
            }
        }
        return _make_environment_map_shading_point(ray, bounce, lod);
    }

    ShadingPoint trace_and_get_shading_point(const RayDesc ray,
                                             const uint instance_mask = 0xff,
                                             const uint16_t bounce = 0,
                                             const LODSampler2D lod = DefaultLODSampler()) {
        return get_shading_point(ray, trace_ray(ray, RAY_FLAG_NONE, instance_mask), bounce, lod);
    }

    // -----------------------------------------

    ShadingPoint _make_environment_map_shading_point(const RayDesc ray,
                                                     const uint16_t bounce,
                                                     const LODSampler2D lod) {
        const float3 pos = ray.Origin + ray.Direction * ENV_MAP_DISTANCE;
        const ShadingData sd =
            ShadingData(-ray.Direction, pos, float2(0), time, OrthonormalFrame(-ray.Direction),
                        -ray.Direction, bounce);

        ShadingPoint sp;
        sp.pos = pos;
        sp.enc_face_normal = encode_normal(-ray.Direction);
        sp.material = env_map.get_sample(sd, material_system.texture_manager, lod);
        return sp;
    }

    void _proceed_ray(inout SceneAccelerationStructure::RayQueryImpl rq) {
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

} // namespace merian
