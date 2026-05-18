import merian_shaders.scene.scene;
import merian_shaders.gbuffer;
import merian_shaders.utils.textures;
import merian_shaders.utils.ray_differential;
import merian_shaders.colors.debug;

namespace merian {

RWTexture2D<float4> emission;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID,
          ParameterBlock<Scene> scene,
          ParameterBlock<WGBuffer> gbuffer) {

    uint2 pixel = tid.xy;
    if (any(pixel >= gbuffer.get_dimensions()))
        return;

    RayDesc ray = scene.camera.generate_ray_pinhole(pixel, gbuffer.get_dimensions());
    Hit hit = scene.trace_ray(ray);

    if (!hit.is_valid()) {
        gbuffer.clear(pixel);
        emission[pixel] = float4(0);
        return;
    }

    GeometryData gd = scene.get_geometry_data(hit.instance_id, hit.geometry_index);
    uint3 indices = scene.get_vertex_indices(gd, hit.primitive_id);

    PackedVertexData v0 = scene.get_packed_vertex_data(gd, indices.x);
    PackedVertexData v1 = scene.get_packed_vertex_data(gd, indices.y);
    PackedVertexData v2 = scene.get_packed_vertex_data(gd, indices.z);

    float3 bary = hit.get_barycentrics3();

    // Interpolate position in object space, then transform
    float3 pos = bary.x * v0.position + bary.y * v1.position + bary.z * v2.position;
    pos = scene.to_world_position(gd, hit.instance_index, pos);

    float2 uv = bary.x * v0.uv + bary.y * v1.uv + bary.z * v2.uv;

    // Face normal from world-space triangle edges; oriented to mesh-canonical front side.
    float3 edge01 = scene.to_world_vector(gd, hit.instance_index, v1.position - v0.position);
    float3 edge02 = scene.to_world_vector(gd, hit.instance_index, v2.position - v0.position);
    float3 face_normal = normalize(cross(edge01, edge02));
    if ((gd.flags & GeometryDataFlags::FrontCounterClockwise) == 0) face_normal = -face_normal;

    // Input shading frame: flat → face normal, smooth → interpolated vertex normal.
    float3 input_normal;
    if (gd.flags & GeometryDataFlags::FlatShading) {
        input_normal = face_normal;
    } else {
        input_normal = scene.get_interpolated_vertex_data(gd, hit.primitive_id, bary).normal;
        input_normal = scene.to_world_normal(gd, hit.instance_index, input_normal);
    }

    // UV-aligned shading frame: per-vertex tangent when the mesh has one, otherwise
    // derived per-triangle from the world-space edges + UV deltas (Lengyel).
    float3 tangent_w;
    float handedness;
    bool tangent_valid;
    if (gd.flags & GeometryDataFlags::HasTangents) {
        const VertexData vd = scene.get_interpolated_vertex_data(gd, hit.primitive_id, bary);
        tangent_w = scene.to_world_vector(gd, hit.instance_index, vd.tangent.xyz);
        handedness = sign(vd.tangent.w);
        tangent_valid = true;
    } else {
        const float2 duv1 = float2(v1.uv) - float2(v0.uv);
        const float2 duv2 = float2(v2.uv) - float2(v0.uv);
        const float det = duv1.x * duv2.y - duv2.x * duv1.y;
        if (abs(det) > 1e-8f) {
            const float winding_sign =
                (gd.flags & GeometryDataFlags::FrontCounterClockwise) ? 1.f : -1.f;
            tangent_w = (duv2.y * edge01 - duv1.y * edge02) / det;
            handedness = sign(det) * winding_sign;
            tangent_valid = true;
        } else {
            tangent_valid = false;
        }
    }

    OrthonormalFrame frame = tangent_valid ? OrthonormalFrame(input_normal, tangent_w, handedness)
                                           : OrthonormalFrame(input_normal);

    // Depth
    float linear_z = length(pos - ray.Origin);

    // Depth gradient: change of linear_z per pixel in x and y
    float3 r_x =
        scene.camera.generate_ray_pinhole(pixel + uint2(1, 0), gbuffer.get_dimensions()).Direction;
    float3 r_y =
        scene.camera.generate_ray_pinhole(pixel + uint2(0, 1), gbuffer.get_dimensions()).Direction;
    float grad_z_x = dot(face_normal, pos - ray.Origin) / dot(face_normal, r_x) - linear_z;
    float grad_z_y = dot(face_normal, pos - ray.Origin) / dot(face_normal, r_y) - linear_z;
    half2 grad_z = half2(grad_z_x, grad_z_y);

    // Delta z: difference in depth from previous frame camera
    float3 prev_object_pos = scene.get_interpolated_prev_object_position(gd, hit.primitive_id, bary);
    float3 prev_pos = scene.to_prev_world_position(gd, hit.instance_index, prev_object_pos);
    float delta_z = length(prev_pos - scene.prev_camera.position) - linear_z;

    // Motion vectors: project previous position onto previous camera.
    // Correct for jitter so accumulated TAA samples land on matching subpixels.
    float2 prev_pixel = scene.prev_camera.project_pinhole(prev_pos, gbuffer.get_dimensions());
    float2 mv = prev_pixel - float2(pixel) + scene.prev_camera.jitter - scene.camera.jitter;

    // Screen-space UV gradients for anisotropic mip selection.
    RayDifferential rd = RayDifferential::from_pinhole_camera(ray.Direction, r_x, r_y);
    rd.propagate(ray.Direction, linear_z, face_normal);
    float2 ddx, ddy;
    rd.compute_uv_gradients(edge01, edge02, face_normal, v0.uv, v1.uv, v2.uv, ddx, ddy);

    let mat = scene.material_system.get_material_sample(
        scene.get_material_id(hit),
        ShadingData(-ray.Direction, pos, uv, scene.time, frame, face_normal, uint16_t(0)),
        GradLODSampler(ddx, ddy));

    // Store
    gbuffer.store_surface(pixel, mat.shading_normal, linear_z, grad_z, delta_z);
    gbuffer.store_hit(pixel, hit);
    gbuffer.store_motion_vectors(pixel, mv);
    gbuffer.store_albedo(pixel, mat.get_bsdf().get_albedo(-ray.Direction));
    emission[pixel] = float4(mat.emission, 0);
}

}
