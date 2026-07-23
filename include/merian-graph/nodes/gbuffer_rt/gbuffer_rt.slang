import merian_shaders.scene.scene;
import merian_shaders.gbuffer;
import merian_shaders.utils.textures;
import merian_shaders.utils.ray_differential;
import merian_shaders.colors.debug;

namespace merian {

RWTexture2D<float4> emission;
// 8-bit TLAS instance cull mask used by the primary ray query. Per-frame from the host.
struct GBufferParams {
    uint instance_mask;
};
ConstantBuffer<GBufferParams> params;

extern static const bool merian_gbuffer_write_emission = true;

[shader("raygeneration")]
void main(ParameterBlock<Scene> scene, ParameterBlock<WGBuffer> gbuffer) {
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 dim = gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    const RayDesc ray = scene.generate_camera_ray(pixel, dim);
    const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, params.instance_mask);

    if (let gd = scene.get_geometry_data(hit)) {
        if ((gd.flags & GeometryDataFlags::UseEnvMap) == 0) {
            const TriangleHit tri = scene.get_triangle_hit(hit, gd);
            const Surface surface = scene.get_surface(hit, tri);
            const float3 prev_pos = scene.get_prev_world_position(hit, gd);

            const float linear_z = length(surface.pos - ray.Origin);
            const float3 r_x = scene.camera.get_pinhole_direction(pixel + uint2(1, 0), dim);
            const float3 r_y = scene.camera.get_pinhole_direction(pixel + uint2(0, 1), dim);
            const half2 grad_z = surface.compute_grad_z(ray, r_x, r_y, linear_z);
            const float delta_z = length(prev_pos - scene.prev_camera.position) - linear_z;
            const float2 mv = scene.compute_motion_vector(prev_pos, surface.pos, pixel, dim);

            // Screen-space UV gradients for anisotropic mip selection.
            const float3 edge01_w =
                scene.to_world_vector(gd, hit.instance_index, tri.get_triangle_edge01());
            const float3 edge02_w =
                scene.to_world_vector(gd, hit.instance_index, tri.get_triangle_edge02());
            RayDifferential rd = RayDifferential::from_pinhole_camera(ray.Direction, r_x, r_y);
            rd.propagate(ray.Direction, linear_z, surface.face_normal);
            float2 ddx, ddy;
            rd.compute_uv_gradients(edge01_w, edge02_w, surface.face_normal, tri.v0.uv, tri.v1.uv,
                                    tri.v2.uv, ddx, ddy);

            OrthonormalFrame frame = surface.frame;
            float3 face_normal = surface.face_normal;
            bool front_facing = true;
            if (gd.flags & GeometryDataFlags::TwoSided) {
                front_facing = dot(-ray.Direction, surface.face_normal) >= 0.f;
                if (!front_facing) {
                    frame.flip_normal();
                    face_normal = -face_normal;
                }
            }

            const ShadingData sd = ShadingData(-ray.Direction, surface.pos, surface.uv, scene.time,
                                               frame, face_normal, uint16_t(0), 1.0f, front_facing);
            let mat = scene.material_system.get_material_sample(surface.material_id, sd,
                                                                GradLODSampler(ddx, ddy));

            gbuffer.store_surface(pixel, mat.shading_normal, linear_z, grad_z, delta_z);
            gbuffer.store_hit(pixel, hit);
            gbuffer.store_motion_vectors(pixel, mv);
            gbuffer.store_albedo(
                pixel, mat.get_bsdf().get_albedo(mat.shading_frame.to_local(-ray.Direction)));
            if (merian_gbuffer_write_emission)
                emission[pixel] = float4(mat.emission, 0);
            return;
        }
    }

    const ShadingPoint sp =
        scene._make_environment_map_shading_point(ray, RenderContext(), DefaultLODSampler());
    const float linear_z = length(sp.pos - ray.Origin);
    const float2 mv = scene.compute_motion_vector(sp.pos, sp.pos, pixel, dim);

    gbuffer.store_surface(pixel, sp.material.shading_normal, linear_z, half2(0), 0);
    gbuffer.store_hit(pixel, hit);
    gbuffer.store_motion_vectors(pixel, mv);
    gbuffer.store_albedo(pixel, Albedo(sp.material.emission));
    if (merian_gbuffer_write_emission)
        emission[pixel] = float4(sp.material.emission, 0);
}

} // namespace merian
