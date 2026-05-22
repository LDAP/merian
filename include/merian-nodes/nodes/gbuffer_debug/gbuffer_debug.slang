import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.shading.materials.gltf_material;
import merian_shaders.utils.textures;
import merian_shaders.colors.debug;

using merian;

struct GBufferDebugBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> output;
};

struct PushConstant {
    int selected_field;
};

[vk::push_constant]
ConstantBuffer<PushConstant> pc;

// Background color for missing / not-applicable data (no hit, cast failure, etc.).
static const float3 MISS = float3(1, 0, 1);

// Resolved per-pixel surface data; lazily built when a scene-dependent field is selected.
struct Surface {
    bool valid;
    Hit hit;
    GeometryData gd;
    PackedVertexData v0, v1, v2;
    float3 bary;
    float3 pos;
    float2 uv;
    float3 edge01, edge02;
    float3 face_normal;
    float3 vertex_normal;
    float3 tangent_w;
    float handedness;
    bool tangent_valid;
    OrthonormalFrame frame;
};

Surface resolve_surface(ParameterBlock<Scene> scene, ParameterBlock<GBuffer> gbuffer, uint2 pixel) {
    Surface s;
    s.hit = gbuffer.get_hit(pixel);
    s.valid = s.hit.is_valid();
    if (!s.valid)
        return s;

    s.gd = scene.get_geometry_data(s.hit.instance_id, s.hit.geometry_index);
    const uint3 indices = scene.get_vertex_indices(s.gd, s.hit.primitive_id);
    s.v0 = scene.get_packed_vertex_data(s.gd, indices.x);
    s.v1 = scene.get_packed_vertex_data(s.gd, indices.y);
    s.v2 = scene.get_packed_vertex_data(s.gd, indices.z);
    s.bary = s.hit.get_barycentrics3();

    const float3 pos_obj = s.bary.x * s.v0.position + s.bary.y * s.v1.position + s.bary.z * s.v2.position;
    s.pos = scene.to_world_position(s.gd, s.hit.instance_index, pos_obj);
    s.uv = s.bary.x * float2(s.v0.uv) + s.bary.y * float2(s.v1.uv) + s.bary.z * float2(s.v2.uv);

    s.edge01 = scene.to_world_vector(s.gd, s.hit.instance_index, s.v1.position - s.v0.position);
    s.edge02 = scene.to_world_vector(s.gd, s.hit.instance_index, s.v2.position - s.v0.position);
    s.face_normal = normalize(cross(s.edge01, s.edge02));
    if (s.gd.flags & GeometryDataFlags::FrontCounterClockwise)
        s.face_normal = -s.face_normal;

    if (s.gd.flags & GeometryDataFlags::FlatShading) {
        s.vertex_normal = s.face_normal;
    } else {
        const VertexData vd = scene.get_interpolated_vertex_data(s.gd, s.hit.primitive_id, s.bary);
        s.vertex_normal = scene.to_world_normal(s.gd, s.hit.instance_index, vd.normal);
    }

    if (s.gd.flags & GeometryDataFlags::HasTangents) {
        const VertexData vd = scene.get_interpolated_vertex_data(s.gd, s.hit.primitive_id, s.bary);
        s.tangent_w = scene.to_world_vector(s.gd, s.hit.instance_index, vd.tangent.xyz);
        s.handedness = sign(vd.tangent.w);
        s.tangent_valid = true;
    } else {
        const float2 duv1 = float2(s.v1.uv) - float2(s.v0.uv);
        const float2 duv2 = float2(s.v2.uv) - float2(s.v0.uv);
        const float det = duv1.x * duv2.y - duv2.x * duv1.y;
        if (abs(det) > 1e-8f) {
            const float winding_sign =
                (s.gd.flags & GeometryDataFlags::FrontCounterClockwise) ? 1.f : -1.f;
            s.tangent_w = (duv2.y * s.edge01 - duv1.y * s.edge02) / det;
            s.handedness = sign(det) * winding_sign;
            s.tangent_valid = true;
        } else {
            s.tangent_w = float3(1, 0, 0);
            s.handedness = 1.f;
            s.tangent_valid = false;
        }
    }

    s.frame = s.tangent_valid ? OrthonormalFrame(s.vertex_normal, s.tangent_w, s.handedness)
                              : OrthonormalFrame(s.vertex_normal);
    return s;
}

MaterialSample sample_material(ParameterBlock<Scene> scene, const Surface s, const uint2 pixel,
                               const uint2 dim) {
    const RayDesc ray = scene.camera.generate_ray_pinhole(pixel, dim);
    const ShadingData sd =
        ShadingData(-ray.Direction, s.pos, s.uv, scene.time, s.frame, s.face_normal, uint16_t(0));
    return scene.material_system.get_material_sample(scene.get_material_id(s.hit), sd);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID,
          ParameterBlock<Scene> scene,
          ParameterBlock<GBufferDebugBinding> params) {

    const uint2 pixel = tid.xy;
    if (not_on_texture(pixel, params.output))
        return;

    const uint2 dim = params.gbuffer.get_dimensions();
    float4 color = float4(0, 0, 0, 1);

    switch (pc.selected_field) {

    // --- Shaded preview ---

    case 0: { // Simple Shading (N dot L, fixed light)
        const float3 n = params.gbuffer.get_normal(pixel);
        const float3 l = normalize(float3(0.4, 0.8, 0.6));
        const float ndotl = saturate(dot(normalize(n), l));
        const float ambient = 0.1f;
        color = float4(float3(ambient + (1.f - ambient) * ndotl), 1);
        break;
    }
    case 1: { // Simple Shading with Albedo
        const float3 n = params.gbuffer.get_normal(pixel);
        const float3 l = normalize(float3(0.4, 0.8, 0.6));
        const float ndotl = saturate(dot(normalize(n), l));
        const float ambient = 0.1f;
        const float3 albedo = params.gbuffer.get_albedo(pixel).reflection;
        color = float4(albedo * (ambient + (1.f - ambient) * ndotl), 1);
        break;
    }
    case 2: { // Albedo
        color = float4(params.gbuffer.get_albedo(pixel).reflection, 1);
        break;
    }

    // --- Normals & tangent frame (world space; Normal Texture is the only tangent-space view) ---

    case 3: { // Normal (shading, post normal-map, world space)
        const float3 n = params.gbuffer.get_normal(pixel);
        color = float4(debug_normal_color(n), 1);
        break;
    }
    case 4: { // Normal Texture (shading normal projected into tangent space)
        const Surface s = resolve_surface(scene, params.gbuffer, pixel);
        if (s.valid) {
            const float3 n_world = normalize(params.gbuffer.get_normal(pixel));
            const float3 n_tangent = s.frame.to_local(n_world);
            color = float4(debug_normal_color(n_tangent), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 5: { // Face Normal (world space)
        const Surface s = resolve_surface(scene, params.gbuffer, pixel);
        color = s.valid ? float4(debug_normal_color(s.face_normal), 1) : float4(MISS, 1);
        break;
    }
    case 6: { // Tangent (world space)
        const Surface s = resolve_surface(scene, params.gbuffer, pixel);
        color = s.valid ? float4(debug_normal_color(s.frame.get_tangent()), 1) : float4(MISS, 1);
        break;
    }
    case 7: { // Bitangent (world space)
        const Surface s = resolve_surface(scene, params.gbuffer, pixel);
        color = s.valid ? float4(debug_normal_color(s.frame.get_bitangent()), 1) : float4(MISS, 1);
        break;
    }
    case 8: { // Tangent W (handedness): white = +1, black = -1
        const Surface s = resolve_surface(scene, params.gbuffer, pixel);
        if (s.valid) {
            const float v = 0.5f + 0.5f * s.handedness;
            color = float4(float3(v), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }

    // --- Material ---

    case 9: { // Alpha
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (hit.is_valid()) {
            const Surface s = resolve_surface(scene, params.gbuffer, pixel);
            const float a = scene.material_system.get_alpha(scene.get_material_id(hit), s.uv);
            color = float4(float3(a), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 10: { // Emissive
        const Surface s = resolve_surface(scene, params.gbuffer, pixel);
        if (s.valid) {
            const MaterialSample mat = sample_material(scene, s, pixel, dim);
            color = float4(mat.emission, 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 11: { // Metallic (GltfMaterial only)
        const Surface s = resolve_surface(scene, params.gbuffer, pixel);
        if (s.valid) {
            const MaterialModel m =
                scene.material_system.get_material(scene.get_material_id(s.hit));
            if (let g = m as GltfMaterial) {
                color = float4(float3(g.get_metallic(scene.material_system.texture_manager, s.uv)),
                               1);
            } else {
                color = float4(MISS, 1);
            }
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 12: { // Roughness (GltfMaterial only)
        const Surface s = resolve_surface(scene, params.gbuffer, pixel);
        if (s.valid) {
            const MaterialModel m =
                scene.material_system.get_material(scene.get_material_id(s.hit));
            if (let g = m as GltfMaterial) {
                color = float4(float3(g.get_roughness(scene.material_system.texture_manager, s.uv)),
                               1);
            } else {
                color = float4(MISS, 1);
            }
        } else {
            color = float4(MISS, 1);
        }
        break;
    }

    // --- IDs & barycentrics ---

    case 13: { // Instance ID
        const uint id = params.gbuffer.get_hit(pixel).instance_id;
        color = (id == uint(-1)) ? float4(MISS, 1) : float4(random_color(id), 1);
        break;
    }
    case 14: { // Geometry Index
        const Hit hit = params.gbuffer.get_hit(pixel);
        color = hit.is_valid() ? float4(random_color(hit.geometry_index), 1) : float4(MISS, 1);
        break;
    }
    case 15: { // Geometry ID
        const Hit hit = params.gbuffer.get_hit(pixel);
        color = hit.is_valid() ? float4(random_color(hit.get_geometry_id()), 1) : float4(MISS, 1);
        break;
    }
    case 16: { // Primitive ID
        const Hit hit = params.gbuffer.get_hit(pixel);
        color = hit.is_valid() ? float4(random_color(hit.primitive_id), 1) : float4(MISS, 1);
        break;
    }
    case 17: { // Material ID
        const Hit hit = params.gbuffer.get_hit(pixel);
        color = hit.is_valid() ? float4(random_color(uint(scene.get_material_id(hit))), 1)
                               : float4(MISS, 1);
        break;
    }
    case 18: { // Barycentrics
        const Hit hit = params.gbuffer.get_hit(pixel);
        color = hit.is_valid() ? float4(hit.get_barycentrics3(), 1) : float4(MISS, 1);
        break;
    }

    // --- Depth & motion ---

    case 19: { // Linear Z (plain)
        color = float4(float3(params.gbuffer.get_linear_z(pixel)), 1);
        break;
    }
    case 20: { // Grad Z (plain, per-axis magnitude)
        const float2 g = params.gbuffer.get_grad_z(pixel);
        color = float4(abs(g), 0, 1);
        break;
    }
    case 21: { // Delta Z (red = negative, blue = positive)
        const float dz = params.gbuffer.get_delta_z(pixel);
        color = float4(max(-dz, 0.f), 0, max(dz, 0.f), 1);
        break;
    }
    case 22: { // Motion Vectors (normalized to fraction of max output dim)
        const float2 m = params.gbuffer.get_motion_vectors(pixel);
        const float s = float(max(dim.x, dim.y));
        color = float4(abs(m) / s, 0, 1);
        break;
    }

    // --- Flags ---

    case 23: { // Flat Shading Flag (1 = flat, 0 = smooth)
        const Surface s = resolve_surface(scene, params.gbuffer, pixel);
        if (s.valid) {
            const float flag = (s.gd.flags & GeometryDataFlags::FlatShading) ? 1.f : 0.f;
            color = float4(float3(flag), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    }

    params.output[pixel] = color;
}
