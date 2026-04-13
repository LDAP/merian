import merian_shaders.scene.scene;
import merian_shaders.gbuffer;

namespace merian {

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          ParameterBlock<Scene> scene,
          ParameterBlock<WGBuffer> gbuffer) {

    uint2 pixel = tid.xy;
    uint2 dims;
    gbuffer.tex0.GetDimensions(dims.x, dims.y);
    if (any(pixel >= dims))
        return;

    RayDesc ray = scene.camera.generate_ray_pinhole(pixel, dims);
    Hit hit = scene.trace_ray(ray);

    if (!hit.is_valid()) {
        gbuffer.clear(pixel);
        return;
    }

    GeometryData gd = scene.get_geometry_data(hit.instance_id, hit.geometry_index);
    GeometryID gid = hit.instance_id + hit.geometry_index;
    uint3 indices = scene.get_vertex_indices(gd.index_buffer_index, hit.primitive_id);

    VertexData v0 = scene.get_vertex_data(gd.vertex_buffer_index, indices.x);
    VertexData v1 = scene.get_vertex_data(gd.vertex_buffer_index, indices.y);
    VertexData v2 = scene.get_vertex_data(gd.vertex_buffer_index, indices.z);

    float3 bary = hit.get_barycentrics3();

    // Interpolate position in object space, then transform
    float3 pos = bary.x * v0.position + bary.y * v1.position + bary.z * v2.position;
    pos = scene.to_world_position(gd, gid, pos);

    // Geometric normal from triangle edges
    float3 edge1 = v1.position - v0.position;
    float3 edge2 = v2.position - v0.position;
    float3 geo_normal = normalize(cross(edge1, edge2));
    geo_normal = scene.to_world_direction(gd, gid, geo_normal);

    // Shading normal (use geometric for now, could interpolate vertex normals)
    float3 shading_normal = geo_normal;

    // Depth
    float linear_z = length(pos - ray.Origin);

    // Depth gradient: change of linear_z per pixel in x and y
    float3 r_x = scene.camera.generate_ray_pinhole(pixel + uint2(1, 0), dims).Direction;
    float3 r_y = scene.camera.generate_ray_pinhole(pixel + uint2(0, 1), dims).Direction;
    float grad_z_x = dot(geo_normal, pos - ray.Origin) / dot(geo_normal, r_x) - linear_z;
    float grad_z_y = dot(geo_normal, pos - ray.Origin) / dot(geo_normal, r_y) - linear_z;
    half2 grad_z = half2(grad_z_x, grad_z_y);

    // Delta z: difference in depth from previous frame camera
    float3 prev_pos = pos; // TODO: use prev_vertex_buffers when available
    float delta_z = length(prev_pos - scene.prev_camera.position) - linear_z;

    // Motion vectors: project previous position onto previous camera
    float2 prev_pixel = scene.prev_camera.project_pinhole(prev_pos, dims);
    float2 mv = prev_pixel - float2(pixel);

    // Store
    gbuffer.store_surface(pixel, shading_normal, linear_z, grad_z, delta_z);
    gbuffer.store_hit(pixel, hit);
    gbuffer.store_motion_vectors(pixel, mv);
}

}
