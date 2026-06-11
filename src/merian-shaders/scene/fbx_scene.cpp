#include "merian-shaders/scene/fbx_scene.hpp"

#include "merian-shaders/shading/materials/pbrt_material.hpp"
#include "merian/utils/normal_encoding.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <ufbx.h>

#include <algorithm>
#include <unordered_map>

namespace merian {

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

namespace {

float3 to_f3(const ufbx_vec3& v) {
    return float3(static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z));
}
float2 to_f2(const ufbx_vec2& v) {
    return float2(static_cast<float>(v.x), static_cast<float>(v.y));
}

float map_real(const ufbx_material_map& m, const float fallback) {
    return m.has_value ? static_cast<float>(m.value_real) : fallback;
}
float3 map_vec3(const ufbx_material_map& m, const float3 fallback) {
    return m.has_value ? to_f3(m.value_vec3) : fallback;
}

// FBX stores Windows-style paths; normalize and probe the common locations.
std::filesystem::path resolve_texture_path(const std::filesystem::path& base_dir,
                                           const ufbx_texture* tex) {
    std::string raw = tex->filename.length > 0
                          ? std::string(tex->filename.data, tex->filename.length)
                          : std::string(tex->relative_filename.data, tex->relative_filename.length);
    std::replace(raw.begin(), raw.end(), '\\', '/');
    const std::filesystem::path rel(raw);

    const std::filesystem::path candidates[] = {
        rel,
        base_dir / rel,
        base_dir / "Textures" / rel.filename(),
    };
    for (const std::filesystem::path& c : candidates) {
        if (std::filesystem::exists(c)) {
            return c;
        }
    }
    return {};
}

// ufbx_matrix is an affine 3x4 in column-major storage (cols[3] is the translation). Merian is
// row-major, so m[row][col] == math(row, col) and the implicit last row is (0, 0, 0, 1).
float4x4 fbx_node_transform(const ufbx_matrix& t) {
    float4x4 m = identity();
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 4; col++) {
            m[row][col] = static_cast<float>(t.cols[col].v[row]);
        }
    }
    return m;
}

} // namespace

// ---------------------------------------------------------------------------

FBXScene::FBXScene(const ShaderCompileContextHandle& compile_context,
                   const ContextHandle& context,
                   const ResourceAllocatorHandle& allocator,
                   const MaterialSystemHandle& material_system)
    : Scene(compile_context, context, allocator, material_system) {

    pbrt_type_id = material_system->register_material_type(PBRT_MATERIAL_SLANG_TYPE_NAME,
                                                           PBRT_MATERIAL_SLANG_MODULE_PATH);
}

FBXScene::~FBXScene() {
    free_scene();
}

void FBXScene::free_scene() {
    if (scene != nullptr) {
        ufbx_free_scene(scene);
        scene = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Material loading
// ---------------------------------------------------------------------------

void FBXScene::load_materials(const CommandBufferHandle& cmd) {
    // Textures upload lazily on first material access.
    texture_slots.assign(scene->textures.count, TextureSlot{});

    // ufbx normalizes every source shader into the same `pbr` map set.
    material_map.resize(scene->materials.count);
    material_flags.assign(scene->materials.count, MeshFlags::IsOpaque);
    bool any_transmissive = false;
    bool any_clearcoat = false;
    bool any_sheen = false;
    for (size_t i = 0; i < scene->materials.count; i++) {
        const ufbx_material* fmat = scene->materials.data[i];
        const ufbx_material_pbr_maps& pbr = fmat->pbr;

        // Legacy FBX Phong/Lambert materials have no real transmission channel; ufbx's Phong->PBR
        // conversion fills transmission anyway (1.0 for opaque assets), so only trust it for PBR
        // shaders.
        const bool pbr_shader = fmat->shader_type != UFBX_SHADER_FBX_PHONG &&
                                fmat->shader_type != UFBX_SHADER_FBX_LAMBERT;
        const float transmission = pbr_shader ? map_real(pbr.transmission_factor, 0.f) : 0.f;
        any_transmissive |= transmission > 0.f;

        PBRTMaterial mat;
        PBRTMaterialPayload& p = mat.payload;

        const auto load = [&](const ufbx_material_map& m, const bool linear) -> TextureID {
            return m.texture != nullptr ? get_or_load_texture(cmd, m.texture, linear)
                                        : TextureID(-1);
        };

        // base color doubles as the alpha-test source (as in glTF/quake).
        p.base_color = map_vec3(pbr.base_color, float3(1)) * map_real(pbr.base_factor, 1.f);
        p.opacity = map_real(pbr.opacity, 1.f);
        mat.header.alpha_texture_id = load(pbr.base_color, false);

        const ufbx_texture* base_tex = pbr.base_color.texture;
        const bool base_has_alpha = base_tex != nullptr &&
                                    base_tex->typed_id < texture_slots.size() &&
                                    texture_slots[base_tex->typed_id].has_alpha;
        const bool opaque = !base_has_alpha && transmission == 0.f && p.opacity >= 0.999f;
        // Back faces are shaded for alpha-cutout (foliage), explicitly double-sided materials, and
        // glass (so the ray hits the exit interface).
        const bool two_sided = base_has_alpha || transmission > 0.f ||
                               fmat->features.features[UFBX_MATERIAL_FEATURE_DOUBLE_SIDED].enabled;
        material_flags[i] = (opaque ? MeshFlags::IsOpaque : MeshFlags::None) |
                            (two_sided ? MeshFlags::TwoSided : MeshFlags::None);

        // metalness / roughness (linear scalar maps; the texture replaces the constant)
        p.metalness_texture = load(pbr.metalness, true);
        p.metalness = p.metalness_texture != TextureID(-1) ? 1.f : map_real(pbr.metalness, 0.f);
        p.roughness_texture = load(pbr.roughness, true);
        p.roughness = p.roughness_texture != TextureID(-1) ? 1.f : map_real(pbr.roughness, 1.f);

        // specular
        p.specular_weight = map_real(pbr.specular_factor, 1.f);
        p.specular_ior = map_real(pbr.specular_ior, 1.5f);

        // emission (sRGB texture × linear factor)
        p.emission = map_vec3(pbr.emission_color, float3(0)) * map_real(pbr.emission_factor, 1.f);
        p.emission_texture = load(pbr.emission_color, false);

        // normal map (linear)
        p.normal_texture = load(pbr.normal_map, true);

        // clearcoat
        p.coat_weight = map_real(pbr.coat_factor, 0.f);
        p.coat_roughness = map_real(pbr.coat_roughness, 0.f);
        p.coat_ior = map_real(pbr.coat_ior, 1.6f);
        any_clearcoat |= p.coat_weight > 0.f;

        // sheen
        p.sheen_weight = map_real(pbr.sheen_factor, 0.f);
        p.sheen_color = map_vec3(pbr.sheen_color, float3(1));
        p.sheen_roughness = map_real(pbr.sheen_roughness, 0.3f);
        any_sheen |= p.sheen_weight > 0.f;

        // transmission (glass)
        p.transmission_weight = transmission;
        p.transmission_color = map_vec3(pbr.transmission_color, float3(1));

        material_map[i] = get_material_system()->add_material(pbrt_type_id, mat);
    }

    // Enable exactly the lobes this asset uses so the Slang compiler folds out the rest.
    // set_enable_* is a no-op when the flag is unchanged.
    get_material_system()->set_enable_transmission(any_transmissive);
    get_material_system()->set_enable_clearcoat(any_clearcoat);
    get_material_system()->set_enable_sheen(any_sheen);

    // Default material for mesh parts without an assigned material.
    PBRTMaterial default_mat;
    default_material_id = get_material_system()->add_material(pbrt_type_id, default_mat);
}

TextureID FBXScene::get_or_load_texture(const CommandBufferHandle& cmd,
                                        const ufbx_texture* tex,
                                        const bool linear) {
    if (tex == nullptr || tex->typed_id >= texture_slots.size()) {
        return TextureID(-1);
    }
    TextureSlot& slot = texture_slots[tex->typed_id];
    TextureID& cached = linear ? slot.id_linear : slot.id_srgb;
    if (cached != TextureID(-1)) {
        return cached;
    }

    const vk::SamplerAddressMode address_mode = tex->wrap_u == UFBX_WRAP_CLAMP
                                                    ? vk::SamplerAddressMode::eClampToEdge
                                                    : vk::SamplerAddressMode::eRepeat;
    // Mipmap color textures only; box-filtering normal/MR maps degrades them.
    const bool generate_mipmaps = !linear;

    TextureHandle texture;
    if (tex->content.size > 0) {
        // Embedded image: decode the in-memory blob.
        int width = 0;
        int height = 0;
        int comp = 0;
        stbi_uc* pixels =
            stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(tex->content.data),
                                  static_cast<int>(tex->content.size), &width, &height, &comp, 4);
        if (pixels == nullptr) {
            SPDLOG_WARN("FBXScene: failed to decode embedded texture '{}'", tex->name.data);
            return TextureID(-1);
        }
        texture = get_allocator()->create_texture_from_rgba8(
            cmd, reinterpret_cast<const uint32_t*>(pixels), static_cast<uint32_t>(width),
            static_cast<uint32_t>(height), address_mode, vk::Filter::eLinear, vk::Filter::eLinear,
            !linear, tex->name.data, generate_mipmaps);
        cmd->barrier(texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
        slot.has_alpha = comp == 4;
        stbi_image_free(pixels);
    } else {
        // External file: one call dispatches to the right host-side loader by extension.
        const std::filesystem::path path = resolve_texture_path(base_dir, tex);
        if (path.empty()) {
            SPDLOG_WARN("FBXScene: texture file not found for '{}'", tex->filename.data);
            return TextureID(-1);
        }
        try {
            bool has_alpha = false;
            texture = get_allocator()->create_texture_from_file(
                cmd, path, /*srgb=*/!linear, address_mode, vk::Filter::eLinear, vk::Filter::eLinear,
                path.filename().string(), generate_mipmaps, &has_alpha);
            slot.has_alpha = has_alpha;
        } catch (const std::exception& e) {
            SPDLOG_WARN("FBXScene: failed to load texture '{}': {}", path.string(), e.what());
            return TextureID(-1);
        }
    }

    cached = get_texture_manager()->add_texture(texture);
    return cached;
}

// ---------------------------------------------------------------------------
// Mesh loading
// ---------------------------------------------------------------------------

void FBXScene::load_meshes() {
    mesh_map.resize(scene->meshes.count);

    for (size_t mesh_index = 0; mesh_index < scene->meshes.count; mesh_index++) {
        const ufbx_mesh* mesh = scene->meshes.data[mesh_index];

        SPDLOG_DEBUG("FBXScene: loading mesh {:>2}/{} {}", mesh_index + 1, scene->meshes.count,
                     mesh->name.data);

        // ufbx is not triangulated; triangulate per face into this scratch buffer.
        std::vector<uint32_t> tri(mesh->max_face_triangles * 3);

        // One merian mesh per material part (a contiguous group of faces sharing a material).
        for (size_t part_index = 0; part_index < mesh->material_parts.count; part_index++) {
            const ufbx_mesh_part& part = mesh->material_parts.data[part_index];
            if (part.num_triangles == 0) {
                continue;
            }

            auto sm = std::make_unique<SimpleMesh>();
            sm->vertices.reserve(part.num_triangles * 3);
            sm->indices.reserve(part.num_triangles);

            // ufbx index -> local vertex index (split attributes give one entry per unique tuple).
            std::unordered_map<uint32_t, uint32_t> remap;
            remap.reserve(part.num_triangles * 3);

            const auto pack = [&](const uint32_t fbx_index) -> uint32_t {
                const auto [it, inserted] =
                    remap.try_emplace(fbx_index, static_cast<uint32_t>(sm->vertices.size()));
                if (inserted) {
                    const float3 pos =
                        to_f3(ufbx_get_vertex_vec3(&mesh->vertex_position, fbx_index));
                    const float3 nrm =
                        mesh->vertex_normal.exists
                            ? normalize(
                                  to_f3(ufbx_get_vertex_vec3(&mesh->vertex_normal, fbx_index)))
                            : float3(0, 1, 0);
                    const float2 uv = mesh->vertex_uv.exists
                                          ? to_f2(ufbx_get_vertex_vec2(&mesh->vertex_uv, fbx_index))
                                          : float2(0, 0);

                    PackedVertexData v;
                    v.position = pos;
                    v.encoded_normal = encode_normal(nrm);
                    // FBX UV origin is bottom-left; Vulkan samples top-left.
                    v.uv = half2(uv.x, 1.0f - uv.y);
                    if (mesh->vertex_tangent.exists) {
                        const float3 tan =
                            to_f3(ufbx_get_vertex_vec3(&mesh->vertex_tangent, fbx_index));
                        float w = 1.f;
                        if (mesh->vertex_bitangent.exists) {
                            const float3 bit =
                                to_f3(ufbx_get_vertex_vec3(&mesh->vertex_bitangent, fbx_index));
                            w = dot(cross(nrm, tan), bit) < 0.f ? -1.f : 1.f;
                        }
                        v.encoded_tangent = encode_tangent(float4(tan, w));
                    } else {
                        v.encoded_tangent = 0u;
                    }
                    sm->vertices.push_back(v);
                }
                return it->second;
            };

            for (size_t f = 0; f < part.face_indices.count; f++) {
                const ufbx_face face = mesh->faces.data[part.face_indices.data[f]];
                const uint32_t num_tris = ufbx_triangulate_face(tri.data(), tri.size(), mesh, face);
                for (uint32_t t = 0; t < num_tris; t++) {
                    // pack() mutates remap, so sequence the calls explicitly.
                    const uint32_t i0 = pack(tri[t * 3 + 0]);
                    const uint32_t i1 = pack(tri[t * 3 + 1]);
                    const uint32_t i2 = pack(tri[t * 3 + 2]);
                    sm->indices.emplace_back(i0, i1, i2);
                }
            }

            // Material for this part (default material is opaque, single-sided).
            MaterialID mat_id = default_material_id;
            MeshFlags flags = MeshFlags::IsOpaque;
            if (part.index < mesh->materials.count) {
                const ufbx_material* m = mesh->materials.data[part.index];
                if (m != nullptr && m->typed_id < material_map.size()) {
                    mat_id = material_map[m->typed_id];
                    flags = material_flags[m->typed_id];
                }
            }

            if (mesh->vertex_tangent.exists) {
                flags = flags | MeshFlags::HasTangents;
            }

            // FBX stores names on nodes, not geometry, so fall back to the instancing node's name.
            std::string base_name;
            if (mesh->name.length > 0) {
                base_name = std::string(mesh->name.data, mesh->name.length);
            } else if (mesh->instances.count > 0 && mesh->instances.data[0]->name.length > 0) {
                const ufbx_string& n = mesh->instances.data[0]->name;
                base_name = std::string(n.data, n.length);
            } else {
                base_name = fmt::format("FBX Mesh {}", mesh_index);
            }
            sm->name = mesh->material_parts.count > 1
                           ? fmt::format("{} ({:02})", base_name, part_index)
                           : base_name;
            sm->material_id = mat_id;
            sm->flags = flags;

            mesh_map[mesh_index].emplace_back(add_mesh(std::move(sm)));
        }
    }
}

// ---------------------------------------------------------------------------
// Scene graph
// ---------------------------------------------------------------------------

void FBXScene::load_node(const ufbx_node* node, const NodeID parent_id) {
    Scene::Node sn;
    sn.name = node->name.length > 0 ? std::string(node->name.data)
                                    : fmt::format("FBX Node {:02}", node->typed_id);
    sn.parent = parent_id;
    sn.local_transform = fbx_node_transform(node->node_to_parent);

    const NodeID nid = add_node(sn);
    node_map[node->typed_id] = nid;

    if (node->mesh != nullptr) {
        for (const MeshID mesh_id : mesh_map[node->mesh->typed_id]) {
            add_mesh_instance(mesh_id, nid);
        }
    }

    for (size_t i = 0; i < node->children.count; i++) {
        load_node(node->children.data[i], nid);
    }
}

void FBXScene::load_cameras() {
    for (size_t i = 0; i < scene->nodes.count; i++) {
        const ufbx_node* node = scene->nodes.data[i];
        if (node->camera == nullptr) {
            continue;
        }
        const ufbx_camera* cam = node->camera;

        // merian row-major: basis in columns 0..2, translation in column 3. With
        // target_camera_axes = right-handed Y-up, FBX cameras look down -Z like glTF.
        const float4x4& mat = get_global_transform(node_map[node->typed_id]);
        const float3 up_vec = normalize(float3(mat[0][1], mat[1][1], mat[2][1]));
        const float3 forward = normalize(float3(mat[0][2], mat[1][2], mat[2][2]));
        const float3 eye = float3(mat[0][3], mat[1][3], mat[2][3]);
        const float3 center = eye - forward;

        const float yfov = radians(static_cast<float>(cam->field_of_view_deg.y));
        const float aspect = cam->aspect_ratio > 0 ? static_cast<float>(cam->aspect_ratio) : 1.f;
        const float znear = cam->near_plane > 0 ? static_cast<float>(cam->near_plane) : 0.01f;
        const float zfar = cam->far_plane > 0 ? static_cast<float>(cam->far_plane) : 1000.f;

        add_camera(std::make_shared<Camera>(eye, center, up_vec, yfov, aspect, znear, zfar));
        SPDLOG_DEBUG("FBXScene: loaded camera '{}' at ({},{},{})", cam->name.data, eye.x, eye.y,
                     eye.z);
    }

    // Fallback: frame the scene if the file has no camera.
    if (get_cameras().empty()) {
        SPDLOG_INFO("FBXScene: no cameras in file, adding default camera");
        add_camera(std::make_shared<Camera>(float3(3, 3, 3), float3(0, 0, 0), get_up(),
                                            radians(60.f), 1920.f / 1080.f, 0.01f, 1000.f));
        AABB& aabb = get_aabb();
        if (aabb.is_valid()) {
            get_active_camera()->look_at_bounding_box(aabb);
        }
    }

    for (const CameraHandle& cam : get_cameras())
        cam->set_jitter_sequence(Camera::JitterSequence::R2);
}

void FBXScene::compute_aabb() {
    AABB& aabb = get_aabb();
    aabb.reset();

    for (size_t mesh_index = 0; mesh_index < scene->meshes.count; mesh_index++) {
        const ufbx_mesh* mesh = scene->meshes.data[mesh_index];
        const ufbx_vec3_list& positions = mesh->vertex_position.values;
        if (positions.count == 0) {
            continue;
        }

        // Local-space AABB of the mesh geometry.
        float3 lo = to_f3(positions.data[0]);
        float3 hi = lo;
        for (size_t v = 1; v < positions.count; v++) {
            const float3 p = to_f3(positions.data[v]);
            lo = min(lo, p);
            hi = max(hi, p);
        }

        // Expand by all eight transformed corners for every instance (the transform may rotate it).
        for (size_t i = 0; i < mesh->instances.count; i++) {
            const NodeID node_id = node_map[mesh->instances.data[i]->typed_id];
            if (node_id == NODE_ID_INVALID) {
                continue;
            }
            const float4x4& transform = get_global_transform(node_id);
            for (int c = 0; c < 8; c++) {
                const float4 corner = float4((c & 1) ? hi.x : lo.x, (c & 2) ? hi.y : lo.y,
                                             (c & 4) ? hi.z : lo.z, 1.f);
                aabb.expand(mul(transform, corner));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

void FBXScene::load(const CommandBufferHandle& cmd, const std::filesystem::path& path) {
    // Defer texture destruction to pool reset so any in-flight frame keeps its bindings valid.
    for (const TextureSlot& slot : texture_slots) {
        for (const TextureID id : {slot.id_srgb, slot.id_linear}) {
            if (id != TextureID(-1)) {
                cmd->keep_until_pool_reset(get_texture_manager()->get_texture(id));
                get_texture_manager()->remove_texture(id);
            }
        }
    }
    texture_slots.clear();
    material_map.clear();
    node_map.clear();
    mesh_map.clear();

    clear_geometry();
    get_material_system()->clear();
    free_scene();

    base_dir = path.parent_path();
    SPDLOG_INFO("FBXScene: loading {}", path.string());

    ufbx_load_opts opts = {};
    // Normalize to the merian convention so transforms and get_up() match the glTF path.
    opts.target_axes = ufbx_axes_right_handed_y_up;
    opts.target_camera_axes = ufbx_axes_right_handed_y_up;
    opts.target_unit_meters = 1.0f;
    opts.space_conversion = UFBX_SPACE_CONVERSION_MODIFY_GEOMETRY;
    // Fold per-mesh geometry transforms into the node graph so node_to_parent is the full
    // transform.
    opts.geometry_transform_handling = UFBX_GEOMETRY_TRANSFORM_HANDLING_HELPER_NODES;
    opts.generate_missing_normals = true;

    ufbx_error error;
    ufbx_scene* parsed = ufbx_load_file(path.string().c_str(), &opts, &error);
    if (parsed == nullptr) {
        // Leave scene null so is_ready() reports false and update() bails out cleanly.
        SPDLOG_ERROR("FBXScene: failed to load '{}': {}", path.string(), error.description.data);
        return;
    }
    scene = parsed;

    // ----------------

    load_materials(cmd);

    load_meshes();

    node_map.assign(scene->nodes.count, NODE_ID_INVALID);
    load_node(scene->root_node, NODE_ID_INVALID);

    // AABB and cameras need the scene graph.
    compute_aabb();

    load_cameras();

    SPDLOG_INFO("FBXScene: loaded '{}' nodes: {}, meshes: {}, materials: {}, textures: {}",
                path.filename().string(), get_scene_graph().size(), get_mesh_infos().size(),
                material_map.size(), scene->textures.count);
}

} // namespace merian
