#pragma once

__exported import merian_shaders.scene.camera;
__exported import merian_shaders.scene.acceleration_structure;
__exported import merian_shaders.scene.environment_map;
__exported import merian_shaders.shading.materials.material_system;

#include "merian-shaders/scene/scene-data.slangh"

namespace merian {

struct Hit {
    uint instance_id;

    uint primitive_id;
    uint geometry_index;

    __init(uint instance_id, uint primitive_id, uint geometry_index) {
        this.instance_id = instance_id;
        this.primitive_id = primitive_id;
        this.geometry_index = geometry_index;
    }

    // invalid hit
    __init() {
        this.instance_id = uint(-1);
        this.primitive_id = uint(-1);
        this.geometry_index = uint(-1);
    }

    bool is_valid() {
        return instance_id != uint(-1);
    }
}

extern static const int merian_scene_index_buffers_count;
extern static const int merian_scene_vertex_buffers_count;
extern static const int merian_scene_prev_vertex_buffers_count;
extern static const int merian_scene_geometry_count;

struct Scene {
    AccelerationStructure as;

    // indexed with instance_id, offset.
    StructuredBuffer<uint3> index_buffers[merian_scene_index_buffers_count];
    // indexed with instance_id, offset.
    StructuredBuffer<VertexData> vertex_buffers[merian_scene_vertex_buffers_count];
    // indexed with instance_id, geometry_index. Only for dynamic meshes only
    StructuredBuffer<PrevVertexData> prev_vertex_buffers[merian_scene_prev_vertex_buffers_count];
    // indexed with (instance_id + geometry_index). The CPU side ensures that instance_ids are
    // offset correctly.
    StructuredBuffer<GeometryData> geometries;

    ParameterBlock<MaterialSystem> material_system;
    EnvMap env_map;

    Camera camera;

    // Global scene parameters
    uint frame;
    float time;
    float time_diff;

    // Global heterogeneous fog parameters
    float3 mu_s;
    float mu_t;

    // -----------------------------------------

    Hit trace_ray(const RayDesc ray,
                  const uint ray_flags = RAY_FLAG_NONE,
                  const uint instances_mask = 0xff) {
        var rq = as.trace_ray(ray, ray_flags, instances_mask);
        _proceed_ray(rq);

        if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            return Hit(rq.CommittedInstanceID(), rq.CommittedPrimitiveIndex(),
                       rq.CommittedGeometryIndex());
        }

        return Hit();
    }

    bool trace_visibility_ray(const RayDesc ray,
                              const uint ray_flags = RAY_FLAG_NONE,
                              const uint instances_mask = 0xff) {
        var rq = as.trace_ray(ray, ray_flags, instances_mask);
        _proceed_ray(rq);

        return rq.CommittedStatus() == COMMITTED_NOTHING;
    }

    GeometryData get_geometry_data(const GeometryID geometry_id) {
        return geometries[geometry_id];
    }

    GeometryData get_geometry_data(const uint instance_id, const uint geometry_index) {
        return get_geometry_data(instance_id + geometry_index);
    }

    MaterialID get_material_id(const GeometryID geometry_id) {
        return get_geometry_data(geometry_id).material_id;
    }

    MaterialID get_material_id(const uint instance_id, const uint geometry_index) {
        return get_geometry_data(instance_id, geometry_index).material_id;
    }

    // Convenience method for get_geometry_data(hit).material_id;
    MaterialID get_material_id(const Hit hit) {
        return get_material_id(hit.instance_id, hit.geometry_index);
    }

    uint3 get_vertex_indices(const uint index_buffer, const uint primitive_id) {
        return index_buffers[index_buffer][primitive_id];
    }

    // interface might change.
    VertexData get_vertex_data(const uint vertex_buffer, const uint vertex_index) {
        return vertex_buffers[vertex_buffer][vertex_index];
    }

    float2 get_vertex_uv(const uint vertex_buffer, const uint vertex_index) {
        return get_vertex_data(vertex_buffer, vertex_index).uv;
    }

    float2 get_uv(const uint index_buffer,
                  const uint primitive_id,
                  const uint vertex_buffer,
                  const float3 barycentrics) {
        const uint3 indices = get_vertex_indices(index_buffer, primitive_id);

        return barycentrics.x * get_vertex_uv(vertex_buffer, indices.x) +
               barycentrics.y * get_vertex_uv(vertex_buffer, indices.y) +
               barycentrics.z * get_vertex_uv(vertex_buffer, indices.z);
    }

    // -----------------------------------------

    void _proceed_ray<RQ : IRayQuery>(inout RQ rq) {
        while (rq.Proceed()) {

            if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                const GeometryData geometry_data =
                    get_geometry_data(rq.CandidateInstanceID(), rq.CandidateGeometryIndex());

                if (material_system.alpha_test(geometry_data.material_id,
                                               get_uv(geometry_data.index_buffer_index,
                                                      rq.CandidatePrimitiveIndex(),
                                                      geometry_data.vertex_buffer_index,
                                                      rq.CandidateRayBarycentrics2()))) {
                    continue;
                }

                rq.CommitNonOpaqueTriangleHit();
            }
        }
    }
}

}
