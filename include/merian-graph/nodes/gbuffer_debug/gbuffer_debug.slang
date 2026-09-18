import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.shading.materials.gltf_material;
import merian_shaders.utils.textures;
import merian_shaders.colors.debug;
import merian_shaders.colors.colorspaces;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;

using merian;

struct GBufferDebugBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> output;
};

struct PushConstant {
    int selected_field;
};

[vk::push_constant] ConstantBuffer<PushConstant> pc;

static const float3 MISS = float3(1, 0, 1);
static const float3 CAST_FAIL = float3(1, 1, 0);

[shader("compute")][numthreads(16, 16, 1)] void main(uint3 tid : SV_DispatchThreadID,
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
        const RayDesc ray = scene.generate_camera_ray(pixel, dim);
        const ShadingPoint sp = scene.get_shading_point(ray, params.gbuffer.get_hit(pixel));
        const float3 l = normalize(float3(0.4, 0.8, 0.6));
        const float ndotl = saturate(dot(normalize(sp.shading_normal), l));
        const float ambient = 0.1f;
        float3 albedo = float3(0);
        if (sp.has_surface_material) {
            albedo = sp.surface_material.get_bsdf()
                         .get_albedo(sp.shading_frame.to_local(-ray.Direction))
                         .reflection;
        }
        color = float4(albedo * (ambient + (1.f - ambient) * ndotl) + sp.emission, 1);
        break;
    }
    case 2: { // Albedo
        color = float4(params.gbuffer.get_albedo(pixel), 1);
        break;
    }

        // --- Normals & tangent frame (world space; Normal Texture is the only tangent-space view)
        // ---

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
            const float h =
                sign(dot(surface.frame.get_bitangent(),
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
            const RayDesc ray = scene.generate_camera_ray(pixel, dim);
            const ShadingPoint sp =
                scene.get_shading_point(ray, hit, gd, RenderContext(), DefaultLODSampler(), false);
            color = float4(sp.emission, 1);
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
                color = float4(
                    float3(g.get_metallic(scene.material_system.texture_manager, tri.get_uv(hit))),
                    1);
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
                color = float4(
                    float3(g.get_roughness(scene.material_system.texture_manager, tri.get_uv(hit))),
                    1);
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

        // --- Lobes & depths ---

    case 24: { // Diffuse Albedo
        color = float4(params.gbuffer.get_diffuse_albedo(pixel), 1);
        break;
    }
    case 25: { // Specular Albedo
        color = float4(params.gbuffer.get_specular_albedo(pixel), 1);
        break;
    }
    case 26: { // Specular Roughness (linear)
        color = float4(float3(params.gbuffer.get_roughness(pixel)), 1);
        break;
    }
    case 27: { // Specular Hit Distance
        color = float4(float3(params.gbuffer.get_specular_hit_distance(pixel)), 1);
        break;
    }
    case 28: { // View Depth (plain)
        color = float4(float3(params.gbuffer.get_view_depth(pixel)), 1);
        break;
    }
    case 29: { // Projected Depth
        color = float4(float3(params.gbuffer.get_projected_depth(pixel)), 1);
        break;
    }

    // What the light grid's cell list is worth where the camera looks: every emitter's
    // unshadowed contribution against the share the cell's slots hold, and against the share the
    // strongest few hold at all.
    case 30: { // Light Grid Coverage
        const Hit hit = params.gbuffer.get_hit(pixel);
        if (!hit.is_valid()) {
            color = float4(MISS, 1);
            break;
        }
        const RayDesc ray = scene.generate_camera_ray(pixel, dim);
        const ShadingPoint sp = scene.get_shading_point(ray, hit);
        if (!sp.has_surface_material) {
            color = float4(MISS, 1);
            break;
        }
        let bsdf = sp.surface_material.get_bsdf();
        const OrthonormalFrame frame = sp.shading_frame;
        const float3 wi_local = frame.to_local(-ray.Direction);
        RandomGenerator rng = PCG(pcg4d16(uint4(pixel, scene.frame, 0)));
        const uint cell = scene.get_light_bucket(sp.pos, rng);
        if (cell == LIGHT_INVALID_INDEX) {
            color = float4(0, 0, 0, 1);
            break;
        }

        float top[LIGHT_GRID_SLOTS];
        float cell_rank[LIGHT_GRID_SLOTS];
        float cell_true[LIGHT_GRID_SLOTS];
        for (uint k = 0; k < LIGHT_GRID_SLOTS; k++) {
            top[k] = 0.f;
            cell_rank[k] = 0.f;
            cell_true[k] = 0.f;
        }

        // A slot names a light source, so the comparison is per source: what it delivers here,
        // and what a cell can rank it by (no receiver, so no BSDF term).
        const uint base = light_bucket_base(cell);
        float total = 0.f;
        float in_cell = 0.f;
        for (uint g = 0; g < scene.nee.table.geometry_count; g++) {
            const LightGeometry lg = scene.nee.table.geometries[g];
            float gw = 0.f;
            float gcw = 0.f;
            for (uint t = 0; t < lg.primitive_count; t++) {
                const uint i = lg.first_triangle + t;
                if (let ls = scene.nee.sample_triangle(i, sp.pos, rng, scene.material_system,
                                                       scene.time)) {
                    const float inv_pdf = 1.f / max(ls.pdf, 1e-30f);
                    const float w =
                        yuv_luminance(bsdf.eval(wi_local, frame.to_local(ls.wo), rng) * ls.Li) *
                        inv_pdf;
                    if (!(w > 0.f))
                        continue;
                    gw += w;
                    gcw += yuv_luminance(ls.Li) * inv_pdf;
                }
            }
            if (!(gw > 0.f))
                continue;
            total += gw;
            for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
                if (scene.nee.grid.slots[base + slot] == g) {
                    in_cell += gw;
                    break;
                }
            }
            if (gw > top[LIGHT_GRID_SLOTS - 1]) {
                uint s = LIGHT_GRID_SLOTS - 1;
                while (s > 0 && top[s - 1] < gw) {
                    top[s] = top[s - 1];
                    s--;
                }
                top[s] = gw;
            }
            if (gcw > cell_rank[LIGHT_GRID_SLOTS - 1]) {
                uint s = LIGHT_GRID_SLOTS - 1;
                while (s > 0 && cell_rank[s - 1] < gcw) {
                    cell_rank[s] = cell_rank[s - 1];
                    cell_true[s] = cell_true[s - 1];
                    s--;
                }
                cell_rank[s] = gcw;
                cell_true[s] = gw;
            }
        }
        if (!(total > 0.f)) {
            color = float4(0, 0, 0, 1);
            break;
        }
        float top_sum = 0.f;
        float cell_best = 0.f;
        for (uint k = 0; k < LIGHT_GRID_SLOTS; k++) {
            top_sum += top[k];
            cell_best += cell_true[k];
        }
        // r: the best LIGHT_GRID_SLOTS sources there are, g: the best a cell's own ranking can
        // pick, b: what the cell holds
        color = float4(max(top_sum / total, 1.f / 255.f), cell_best / total, in_cell / total, 1);
        break;
    }
    }

    params.output[pixel] = color;
}
