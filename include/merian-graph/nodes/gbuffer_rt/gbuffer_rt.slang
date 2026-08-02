import merian_shaders.scene.scene;
import merian_shaders.gbuffer;
import merian_shaders.utils.textures;
import merian_shaders.utils.encoding;
import merian_shaders.utils.ray_differential;
import merian_shaders.colors.debug;

namespace merian {

RWTexture2D<float4> emission;
// Packed primary ShadingPoint, 3x uint4 per pixel: {pos, enc_face_normal},
// {prev_pos, enc_shading_normal}, {enc_tangent, pack_bsdf(), 0}. Emission is not stored.
RWStructuredBuffer<uint4> primary_samples;
// 8-bit TLAS instance cull mask used by the primary ray query. Per-frame from the host.
struct GBufferParams {
    uint instance_mask;
};
ConstantBuffer<GBufferParams> params;

extern static const bool merian_gbuffer_write_emission = true;
extern static const bool merian_gbuffer_write_primary = false;

void store_primary_sample(const uint2 pixel, const uint2 dim, const float3 pos,
                          const float3 prev_pos, const uint enc_face_normal,
                          const MaterialSample mat) {
    const OrthonormalFrame frame = mat.shading_frame;
    const float3 t = frame.get_tangent();
    const float hand =
        dot(cross(frame.get_normal(), t), frame.get_bitangent()) < 0.f ? -1.f : 1.f;
    const uint base = 3 * (pixel.x + pixel.y * dim.x);
    primary_samples[base + 0] = uint4(asuint(pos), enc_face_normal);
    primary_samples[base + 1] = uint4(asuint(prev_pos), encode_normal(frame.get_normal()));
    primary_samples[base + 2] = uint4(encode_tangent(float4(t, hand)), mat.pack_bsdf(), 0);
}

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
            if (merian_gbuffer_write_primary)
                store_primary_sample(pixel, dim, surface.pos, prev_pos,
                                     encode_normal(surface.face_normal), mat);
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
    if (merian_gbuffer_write_primary)
        store_primary_sample(pixel, dim, sp.pos, sp.prev_pos, sp.enc_face_normal, sp.material);
}

} // namespace merian
