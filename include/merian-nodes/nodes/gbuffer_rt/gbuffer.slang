import merian_shaders.scene.scene;
import merian_shaders.utils.encoding;

namespace merian {

struct GBufferOutput {
    // .xyz = world position, .w = hit distance (< 0 = miss)
    RWTexture2D<float4> position;
    // .xyz = world normal (decoded), .w = 0
    RWTexture2D<float4> normal;
    // .xyzw = base color (albedo)
    RWTexture2D<float4> albedo;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID,
          ParameterBlock<Scene> scene,
          ParameterBlock<GBufferOutput> gbuffer) {

    uint2 pixel = tid.xy;
    uint2 dims;
    gbuffer.position.GetDimensions(dims.x, dims.y);
    if (any(pixel >= dims))
        return;

    RayDesc ray = scene.camera.generate_ray_pinhole(pixel, dims);
    Hit hit = scene.trace_ray(ray);

    if (!hit.is_valid()) {
        // miss: blue-ish background so we can tell misses from errors
        gbuffer.position[pixel] = float4(0, 0, 0, -1);
        gbuffer.normal[pixel] = float4(0.1, 0.1, 0.3, 0);
        gbuffer.albedo[pixel] = float4(0.1, 0.1, 0.3, 0);
        return;
    }

    GeometryData gd = scene.get_geometry_data(hit.instance_id, hit.geometry_index);
    uint3 indices = scene.get_vertex_indices(gd.index_buffer_index, hit.primitive_id);

    VertexData v0 = scene.get_vertex_data(gd.vertex_buffer_index, indices.x);
    VertexData v1 = scene.get_vertex_data(gd.vertex_buffer_index, indices.y);
    VertexData v2 = scene.get_vertex_data(gd.vertex_buffer_index, indices.z);

    float3 bary = hit.get_barycentrics3();

    float3 pos = bary.x * v0.position + bary.y * v1.position + bary.z * v2.position;

    // Flat geometric normal from triangle edges
    float3 edge1 = v1.position - v0.position;
    float3 edge2 = v2.position - v0.position;
    float3 normal = normalize(cross(edge1, edge2));

    gbuffer.position[pixel] = float4(pos, length(pos - ray.Origin));
    gbuffer.normal[pixel] = float4(normal * 0.5 + 0.5, 1);
    gbuffer.albedo[pixel] = float4(1, 1, 1, 1);
}

}
