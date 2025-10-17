#pragma once

__exported import merian_shaders.scene.camera;
__exported import merian_shaders.scene.acceleration_structure;
__exported import merian_shaders.scene.environment_map;
__exported import merian_shaders.shading.materials.material_system;

namespace merian {

struct PackedVertexData {
    float3 position;
    half2 uv;
}

struct PrevVertexData {
    float3 position;
}

struct Hit {
    uint instance_id;
    uint primitive_id;

    __init(uint instance_id, uint primitive_id) {
        this.instance_id = instance_id;
        this.primitive_id = primitive_id;
    }

    // invalid hit
    __init() {
        this.instance_id = uint(-1);
        this.primitive_id = uint(-1);
    }

    bool is_valid() {
        return instance_id != uint(-1);
    }
}

extern static const int merian_scene_index_buffers_count;
extern static const int merian_scene_vertex_buffers_count;
extern static const int merian_scene_prev_vertex_buffers_count;

struct Scene {
    AccelerationStructure as;

    StructuredBuffer<uint3> index_buffers[merian_scene_index_buffers_count];
    StructuredBuffer<PackedVertexData> vertex_buffers[merian_scene_vertex_buffers_count];
    // only for dynamic meshes only
    StructuredBuffer<PrevVertexData> prev_vertex_buffers[merian_scene_prev_vertex_buffers_count];

    ParameterBlock<MaterialSystem> material_system;
    EnvMap env_map;

    Camera camera;

    // -----------------------------------------

    Hit trace_ray(const RayDesc ray,
                  const uint ray_flags = RAY_FLAG_NONE,
                  const uint instances_mask = 0xff) {
        var rq = as.trace_ray(ray, ray_flags, instances_mask);
        _proceed_ray(rq);

        if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            return Hit(rq.CommittedInstanceID(), rq.CommittedPrimitiveIndex());
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

    // -----------------------------------------

    void _proceed_ray<RQ : IRayQuery>(inout RQ rq) {
        while (rq.Proceed()) {

            if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                if (material_system.alpha_test()) {
                    continue;
                }

                rq.CommitNonOpaqueTriangleHit();
            }
        }
    }
}

}
