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

static const float3 MISS = float3(1, 0, 1);
static const float3 CAST_FAIL = float3(1, 1, 0);

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
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (let gd = scene.get_geometry_data(hit)) {
            const TriangleHit tri = scene.get_triangle_hit(hit, gd);
            const Surface surface = scene.get_surface(hit, tri);
            const float3 n_world = normalize(params.gbuffer.get_normal(pixel));
            const float3 n_tangent = surface.frame.to_local(n_world);
            color = float4(debug_normal_color(n_tangent), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 5: { // Face Normal (world space)
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (let gd = scene.get_geometry_data(hit)) {
            const TriangleHit tri = scene.get_triangle_hit(hit, gd);
            const Surface surface = scene.get_surface(hit, tri);
            color = float4(debug_normal_color(surface.face_normal), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 6: { // Tangent (world space)
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (let gd = scene.get_geometry_data(hit)) {
            const TriangleHit tri = scene.get_triangle_hit(hit, gd);
            const Surface surface = scene.get_surface(hit, tri);
            color = float4(debug_normal_color(surface.frame.get_tangent()), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 7: { // Bitangent (world space)
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (let gd = scene.get_geometry_data(hit)) {
            const TriangleHit tri = scene.get_triangle_hit(hit, gd);
            const Surface surface = scene.get_surface(hit, tri);
            color = float4(debug_normal_color(surface.frame.get_bitangent()), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 8: { // Tangent W (handedness): white = +1, black = -1
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (let gd = scene.get_geometry_data(hit)) {
            const TriangleHit tri = scene.get_triangle_hit(hit, gd);
            const Surface surface = scene.get_surface(hit, tri);
            const float h = sign(dot(surface.frame.get_bitangent(),
                                     cross(surface.frame.get_normal(), surface.frame.get_tangent())));
            color = float4(float3(0.5f + 0.5f * h), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }

    // --- Material ---

    case 9: { // Alpha
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (let gd = scene.get_geometry_data(hit)) {
            const TriangleHit tri = scene.get_triangle_hit(hit, gd);
            const float a = scene.material_system.get_alpha(gd.material_id, tri.get_uv(hit));
            color = float4(float3(a), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 10: { // Emissive
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (let gd = scene.get_geometry_data(hit)) {
            const RayDesc ray = scene.camera.generate_ray_pinhole(pixel, dim);
            const ShadingPoint sp = scene.get_shading_point(ray, hit, gd);
            color = float4(sp.material.emission, 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 11: { // Metallic (GltfMaterial only)
        // https://github.com/shader-slang/slang/issues/11266
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (let gd = scene.get_geometry_data(hit)) {
            const TriangleHit tri = scene.get_triangle_hit(hit, gd);
            const MaterialModel m = scene.material_system.get_material(gd.material_id);
            if (let g = m as GltfMaterial) {
                color = float4(float3(g.get_metallic(scene.material_system.texture_manager, tri.get_uv(hit))), 1);
            } else {
                color = float4(CAST_FAIL, 1);
            }
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    case 12: { // Roughness (GltfMaterial only)
        // https://github.com/shader-slang/slang/issues/11266
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (let gd = scene.get_geometry_data(hit)) {
            const TriangleHit tri = scene.get_triangle_hit(hit, gd);
            const MaterialModel m = scene.material_system.get_material(gd.material_id);
            if (let g = m as GltfMaterial) {
                color = float4(float3(g.get_roughness(scene.material_system.texture_manager, tri.get_uv(hit))), 1);
            } else {
                color = float4(CAST_FAIL, 1);
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
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (let gd = scene.get_geometry_data(hit)) {
            const float flag = (gd.flags & GeometryDataFlags::FlatShading) ? 1.f : 0.f;
            color = float4(float3(flag), 1);
        } else {
            color = float4(MISS, 1);
        }
        break;
    }
    }

    params.output[pixel] = color;
}
