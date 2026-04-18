#include "merian-shaders/scene/gltf_scene.hpp"

#include <fmt/format.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <spdlog/spdlog.h>
#include <tiny_gltf.h>

namespace merian {

GLTFScene::GLTFScene(const ShaderCompileContextHandle& compile_context,
                     const ContextHandle& context,
                     const ResourceAllocatorHandle& allocator,
                     const ShaderObjectAllocatorHandle& obj_allocator,
                     const MaterialSystemHandle& material_system)
    : Scene(compile_context, context, allocator, obj_allocator, material_system) {

    diffuse_type_id = material_system->register_material_type(
        "merian::DiffuseMaterial", "merian-shaders/shading/materials/diffuse-material.slang");
}

GLTFScene::~GLTFScene() = default;

// ---------------------------------------------------------------------------
// Accessor helpers
// ---------------------------------------------------------------------------

namespace {

const uint8_t* get_accessor_base(const tinygltf::Model& model, int accessor_index) {
    const auto& accessor = model.accessors[accessor_index];
    const auto& bv = model.bufferViews[accessor.bufferView];
    const auto& buf = model.buffers[bv.buffer];
    return buf.data.data() + bv.byteOffset + accessor.byteOffset;
}

int get_accessor_byte_stride(const tinygltf::Model& model, int accessor_index) {
    const auto& accessor = model.accessors[accessor_index];
    const auto& bv = model.bufferViews[accessor.bufferView];
    return accessor.ByteStride(bv);
}

template <typename T> T read_strided(const uint8_t* base, int byte_stride, uint32_t index) {
    return *reinterpret_cast<const T*>(base + byte_stride * index);
}

// Mesh referencing buffer data in a tinygltf::Model (owned by GLTFScene).
class GLTFMesh : public Mesh {
  public:
    const uint8_t* pos_base = nullptr;
    int pos_stride = 0;
    const uint8_t* nrm_base = nullptr;
    int nrm_stride = 0;
    const uint8_t* uv_base = nullptr;
    int uv_stride = 0;
    const uint8_t* tan_base = nullptr;
    int tan_stride = 0;
    uint32_t vertex_count = 0;

    // null -> sequential indices
    const uint8_t* idx_base = nullptr;
    int idx_component_type = 0; // TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE / _UNSIGNED_SHORT / _UNSIGNED_INT
    uint32_t primitive_count = 0;

    uint32_t get_vertex_count() const override {
        return vertex_count;
    }
    uint32_t get_primitive_count() const override {
        return primitive_count;
    }

    float3 get_position(uint32_t v) const override {
        return read_strided<float3>(pos_base, pos_stride, v);
    }
    float3 get_normal(uint32_t v) const override {
        if (nrm_base == nullptr)
            return float3(0, 1, 0);
        return read_strided<float3>(nrm_base, nrm_stride, v);
    }
    float2 get_uv(uint32_t v) const override {
        if (uv_base == nullptr)
            return float2(0, 0);
        return read_strided<float2>(uv_base, uv_stride, v);
    }
    float4 get_tangent(uint32_t v) const override {
        if (tan_base == nullptr)
            return float4(1, 0, 0, 1);
        return read_strided<float4>(tan_base, tan_stride, v);
    }

    uint3 get_indices(uint32_t p) const override {
        if (idx_base == nullptr) {
            return uint3(p * 3, p * 3 + 1, p * 3 + 2);
        }
        if (idx_component_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            const auto* src = reinterpret_cast<const uint16_t*>(idx_base);
            return uint3(src[p * 3], src[p * 3 + 1], src[p * 3 + 2]);
        }
        if (idx_component_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            const auto* src = reinterpret_cast<const uint32_t*>(idx_base);
            return uint3(src[p * 3], src[p * 3 + 1], src[p * 3 + 2]);
        }
        // TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
        return uint3(idx_base[p * 3], idx_base[p * 3 + 1], idx_base[p * 3 + 2]);
    }
};

float4x4 gltf_node_transform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        // glTF stores a 4x4 matrix column-major: node.matrix[col * 4 + row]
        // holds the math element (row, col). Merian is row-major, so
        // m[row][col] == math(row, col).
        float4x4 m;
        for (int col = 0; col < 4; col++)
            for (int row = 0; row < 4; row++)
                m[row][col] = static_cast<float>(node.matrix[col * 4 + row]);
        return m;
    }

    float4x4 T = identity();
    float4x4 R = identity();
    float4x4 S = identity();

    if (node.translation.size() == 3) {
        // merian is row-major: translation in last column
        T[0][3] = static_cast<float>(node.translation[0]);
        T[1][3] = static_cast<float>(node.translation[1]);
        T[2][3] = static_cast<float>(node.translation[2]);
    }

    if (node.rotation.size() == 4) {
        // glTF quaternion: [x, y, z, w]
        glm::quat q(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]),
                    static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2]));
        glm::mat4 rm = glm::mat4_cast(q);
        // glm stores the rotation column-major; transpose into merian row-major
        // so that R[row][col] == math(row, col).
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                R[i][j] = rm[j][i];
    }

    if (node.scale.size() == 3) {
        S[0][0] = static_cast<float>(node.scale[0]);
        S[1][1] = static_cast<float>(node.scale[1]);
        S[2][2] = static_cast<float>(node.scale[2]);
    }

    return mul(mul(T, R), S);
}

} // namespace

// ---------------------------------------------------------------------------
// Material loading
// ---------------------------------------------------------------------------

namespace {

vk::Filter gltf_mag_filter_to_vk(int mag_filter) {
    switch (mag_filter) {
    case TINYGLTF_TEXTURE_FILTER_NEAREST:
        return vk::Filter::eNearest;
    case TINYGLTF_TEXTURE_FILTER_LINEAR:
    default:
        return vk::Filter::eLinear;
    }
}

// Translate the glTF minFilter into the matching vk::Filter, mipmap mode and a "wants mipmaps"
// bit. glTF couples all three via the *_MIPMAP_* enum values.
void gltf_min_filter_to_vk(int min_filter,
                           vk::Filter& out_filter,
                           vk::SamplerMipmapMode& out_mipmap_mode,
                           bool& out_wants_mipmaps) {
    switch (min_filter) {
    case TINYGLTF_TEXTURE_FILTER_NEAREST:
        out_filter = vk::Filter::eNearest;
        out_mipmap_mode = vk::SamplerMipmapMode::eNearest;
        out_wants_mipmaps = false;
        break;
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
        out_filter = vk::Filter::eNearest;
        out_mipmap_mode = vk::SamplerMipmapMode::eNearest;
        out_wants_mipmaps = true;
        break;
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
        out_filter = vk::Filter::eNearest;
        out_mipmap_mode = vk::SamplerMipmapMode::eLinear;
        out_wants_mipmaps = true;
        break;
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
        out_filter = vk::Filter::eLinear;
        out_mipmap_mode = vk::SamplerMipmapMode::eNearest;
        out_wants_mipmaps = true;
        break;
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
        out_filter = vk::Filter::eLinear;
        out_mipmap_mode = vk::SamplerMipmapMode::eLinear;
        out_wants_mipmaps = true;
        break;
    case TINYGLTF_TEXTURE_FILTER_LINEAR:
    default:
        out_filter = vk::Filter::eLinear;
        out_mipmap_mode = vk::SamplerMipmapMode::eLinear;
        out_wants_mipmaps = false;
        break;
    }
}

vk::SamplerAddressMode gltf_wrap_to_vk(int wrap) {
    switch (wrap) {
    case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
        return vk::SamplerAddressMode::eClampToEdge;
    case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
        return vk::SamplerAddressMode::eMirroredRepeat;
    case TINYGLTF_TEXTURE_WRAP_REPEAT:
    default:
        return vk::SamplerAddressMode::eRepeat;
    }
}

} // namespace

void GLTFScene::load_materials(const CommandBufferHandle& cmd) {
    const auto& sampler_pool = get_allocator()->get_sampler_pool();

    // Pre-acquire one VkSampler per glTF sampler. We use SamplerPool::acquire_sampler (instead of
    // for_filter_and_address_mode) because glTF specifies wrapS / wrapT independently and the
    // convenience helper only supports a single address mode for all axes.
    gltf_samplers.clear();
    gltf_samplers.reserve(model->samplers.size());
    gltf_sampler_wants_mipmaps.assign(model->samplers.size(), false);
    for (uint32_t i = 0; i < model->samplers.size(); i++) {
        const auto& s = model->samplers[i];

        vk::Filter mag = vk::Filter::eLinear;
        vk::Filter min = vk::Filter::eLinear;
        vk::SamplerMipmapMode mipmap_mode = vk::SamplerMipmapMode::eLinear;
        bool wants_mipmaps = false;

        if (s.magFilter >= 0) {
            mag = gltf_mag_filter_to_vk(s.magFilter);
        }
        if (s.minFilter >= 0) {
            gltf_min_filter_to_vk(s.minFilter, min, mipmap_mode, wants_mipmaps);
        }

        const vk::SamplerCreateInfo info{
            {},
            mag,
            min,
            mipmap_mode,
            gltf_wrap_to_vk(s.wrapS),
            gltf_wrap_to_vk(s.wrapT),
            vk::SamplerAddressMode::eRepeat, // wrapR — irrelevant for 2D textures
            0.f,
            VK_FALSE, // anisotropy off (glTF 2.0 doesn't specify it)
            0.f,
            VK_FALSE,
            vk::CompareOp::eNever,
            0.f,
            VK_LOD_CLAMP_NONE,
        };
        gltf_samplers.push_back(sampler_pool->acquire_sampler(info));
        gltf_sampler_wants_mipmaps[i] = wants_mipmaps;
    }

    // Default sampler for textures without a sampler reference (linear, repeat, no mipmaps).
    default_gltf_sampler =
        sampler_pool->for_filter_and_address_mode(vk::Filter::eLinear, vk::Filter::eLinear,
                                                  vk::SamplerAddressMode::eRepeat,
                                                  vk::SamplerMipmapMode::eLinear, false);

    // Reset slot table; textures upload lazily on first material access.
    texture_slots.assign(model->textures.size(), GltfTextureSlot{});

    // Build materials. get_or_load_texture pulls in only the textures actually referenced.
    material_map.resize(model->materials.size());
    for (size_t i = 0; i < model->materials.size(); i++) {
        const auto& gmat = model->materials[i];
        const auto& pbr = gmat.pbrMetallicRoughness;

        DiffuseMaterial mat;
        mat.base_color_factor = float4(
            static_cast<float>(pbr.baseColorFactor[0]), static_cast<float>(pbr.baseColorFactor[1]),
            static_cast<float>(pbr.baseColorFactor[2]), static_cast<float>(pbr.baseColorFactor[3]));

        // baseColor is an sRGB color texture (glTF spec, "PBR Methodology").
        if (pbr.baseColorTexture.index >= 0) {
            mat.header.alpha_texture_id =
                get_or_load_texture(cmd, pbr.baseColorTexture.index, /*linear=*/false);
        }

        material_map[i] = get_material_system()->add_material(diffuse_type_id, mat);
    }

    // Default material for primitives without one
    if (material_map.empty()) {
        DiffuseMaterial default_mat;
        material_map.push_back(get_material_system()->add_material(diffuse_type_id, default_mat));
    }
}

TextureID GLTFScene::get_or_load_texture(const CommandBufferHandle& cmd,
                                         const int gltf_tex_idx,
                                         const bool linear) {
    if (gltf_tex_idx < 0 || gltf_tex_idx >= static_cast<int>(texture_slots.size())) {
        return TextureID(-1);
    }

    GltfTextureSlot& slot = texture_slots[gltf_tex_idx];
    TextureID& cached = linear ? slot.id_linear : slot.id_srgb;
    if (cached != TextureID(-1)) {
        return cached;
    }

    const auto& tex = model->textures[gltf_tex_idx];

    if (tex.source < 0 || tex.source >= static_cast<int>(model->images.size())) {
        SPDLOG_WARN("GLTFScene: texture {} has no source image", gltf_tex_idx);
        return TextureID(-1);
    }
    const auto& img = model->images[tex.source];
    if (img.image.empty() || img.width <= 0 || img.height <= 0) {
        SPDLOG_WARN("GLTFScene: skipping invalid image '{}'", img.name);
        return TextureID(-1);
    }
    // tinygltf decodes images to RGBA 8-bit by default (component == 4)
    if (img.component != 4 || img.bits != 8) {
        SPDLOG_WARN("GLTFScene: unsupported image format ({} components, {} bits) for '{}'",
                    img.component, img.bits, img.name);
        return TextureID(-1);
    }

    SamplerHandle sampler;
    bool wants_mipmaps = false;
    if (tex.sampler >= 0 && tex.sampler < static_cast<int>(gltf_samplers.size())) {
        sampler = gltf_samplers[tex.sampler];
        wants_mipmaps = gltf_sampler_wants_mipmaps[tex.sampler];
    } else {
        sampler = default_gltf_sampler;
    }

    // Mipmap policy:
    //  - sRGB color textures: honor the glTF sampler, plus opt-in `force_mipmaps_color`.
    //  - linear data textures (normal/MR/occlusion): honor the glTF sampler, but warn — naive
    //    box-filter mipmapping degrades these (normals lose normalization, roughness/metalness/
    //    occlusion need specialized filtering).
    bool generate_mipmaps;
    if (linear) {
        if (wants_mipmaps) {
            SPDLOG_WARN("GLTFScene: texture {} ('{}') is sampled as a normal/MR/occlusion map but "
                        "its sampler requests *_MIPMAP_*; generating mipmaps anyway, expect "
                        "filtering artifacts (specialized mipmap generation is not implemented)",
                        gltf_tex_idx, img.name);
        }
        generate_mipmaps = wants_mipmaps;
    } else {
        generate_mipmaps = wants_mipmaps || force_mipmaps_color;
    }

    SPDLOG_DEBUG("GLTFScene: uploading texture {} (image '{}'), linear={}, mips={}", gltf_tex_idx,
                 img.name, linear, generate_mipmaps);

    TextureHandle texture = get_allocator()->create_texture_from_rgba8(
        cmd, reinterpret_cast<const uint32_t*>(img.image.data()),
        static_cast<uint32_t>(img.width), static_cast<uint32_t>(img.height), sampler, !linear,
        img.name, generate_mipmaps);
    cmd->barrier(texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));

    cached = get_texture_manager()->add_texture(texture);
    return cached;
}

void GLTFScene::load_meshes() {
    mesh_map.resize(model->meshes.size());

    for (uint32_t gltf_mesh_id = 0; gltf_mesh_id < model->meshes.size(); gltf_mesh_id++) {
        const auto& gmesh = model->meshes[gltf_mesh_id];

        SPDLOG_DEBUG("GLFWScene: loading mesh {:>2}/{} {}", gltf_mesh_id + 1, model->meshes.size(),
                     gmesh.name);

        for (uint32_t primitive_index = 0; primitive_index < gmesh.primitives.size();
             primitive_index++) {
            const auto& prim = gmesh.primitives[primitive_index];

            if (prim.mode != -1 && prim.mode != TINYGLTF_MODE_TRIANGLES)
                continue;

            auto pos_it = prim.attributes.find("POSITION");
            if (pos_it == prim.attributes.end())
                continue;

            auto mesh = std::make_unique<GLTFMesh>();

            const auto& pos_acc = model->accessors[pos_it->second];
            mesh->vertex_count = static_cast<uint32_t>(pos_acc.count);
            mesh->pos_base = get_accessor_base(*model, pos_it->second);
            mesh->pos_stride = get_accessor_byte_stride(*model, pos_it->second);

            if (auto it = prim.attributes.find("NORMAL"); it != prim.attributes.end()) {
                mesh->nrm_base = get_accessor_base(*model, it->second);
                mesh->nrm_stride = get_accessor_byte_stride(*model, it->second);
            }
            if (auto it = prim.attributes.find("TEXCOORD_0"); it != prim.attributes.end()) {
                mesh->uv_base = get_accessor_base(*model, it->second);
                mesh->uv_stride = get_accessor_byte_stride(*model, it->second);
            }
            if (auto it = prim.attributes.find("TANGENT"); it != prim.attributes.end()) {
                mesh->tan_base = get_accessor_base(*model, it->second);
                mesh->tan_stride = get_accessor_byte_stride(*model, it->second);
            }

            if (prim.indices >= 0) {
                const auto& idx_acc = model->accessors[prim.indices];
                mesh->idx_base = get_accessor_base(*model, prim.indices);
                mesh->idx_component_type = idx_acc.componentType;
                mesh->primitive_count = static_cast<uint32_t>(idx_acc.count / 3);
            } else {
                mesh->primitive_count = mesh->vertex_count / 3;
            }

            // Determine material
            MaterialID mat_id = material_map[0]; // default
            if (prim.material >= 0 && prim.material < static_cast<int>(material_map.size())) {
                mat_id = material_map[prim.material];
            }

            // Determine flags
            GeometryFlags flags = GeometryFlags::None;
            if (prim.material >= 0 && prim.material < static_cast<int>(model->materials.size())) {
                if (model->materials[prim.material].alphaMode == "OPAQUE") {
                    flags = flags | GeometryFlags::IsOpaque;
                }
            } else {
                flags = flags | GeometryFlags::IsOpaque;
            }

            mesh->name =
                gmesh.primitives.size() > 1
                    ? fmt::format("{} ({:02})",
                                  gmesh.name.empty() ? fmt::format("GLTF Mesh {}", gltf_mesh_id)
                                                     : gmesh.name,
                                  primitive_index)
                    : gmesh.name;
            mesh->material_id = mat_id;
            mesh->flags = flags;

            const MeshID mesh_id = add_mesh(std::move(mesh));
            mesh_map[gltf_mesh_id].emplace_back(mesh_id);
        }
    }
}

void GLTFScene::load_node(int gltf_node_index, NodeID parent_id) {
    const auto& gnode = model->nodes[gltf_node_index];

    SceneNode sn;
    sn.name = gnode.name.empty() ? fmt::format("GLTF Node {:02}", gltf_node_index) : gnode.name;
    sn.parent = parent_id;
    sn.local_transform = gltf_node_transform(gnode);

    NodeID nid = add_node(sn);
    node_map[gltf_node_index] = nid;

    if (gnode.mesh >= 0) {
        for (const MeshID mesh_id : mesh_map[gnode.mesh]) {
            add_mesh_instance(mesh_id, nid);
        }
    }

    // Recurse into children
    for (int child_idx : gnode.children) {
        load_node(child_idx, nid);
    }
}

void GLTFScene::load_cameras() {
    for (size_t ni = 0; ni < model->nodes.size(); ni++) {
        const auto& gnode = model->nodes[ni];
        if (gnode.camera < 0 || gnode.camera >= static_cast<int>(model->cameras.size()))
            continue;

        const auto& gcam = model->cameras[gnode.camera];
        NodeID nid = node_map[ni];

        // merian row-major: basis in columns 0..2, translation in column 3
        const float4x4& mat = get_global_transform(nid);
        float3 up_vec = normalize(float3(mat[0][1], mat[1][1], mat[2][1]));
        float3 forward = normalize(float3(mat[0][2], mat[1][2], mat[2][2]));
        float3 eye = float3(mat[0][3], mat[1][3], mat[2][3]);

        // glTF cameras look down -Z in local space
        float3 center = eye - forward;

        if (gcam.type == "perspective") {
            float fov = 60.f;
            float aspect = 1.f;
            float znear = 0.01f;
            float zfar = 1000.f;

            fov = static_cast<float>(glm::degrees(gcam.perspective.yfov));
            if (gcam.perspective.aspectRatio > 0)
                aspect = static_cast<float>(gcam.perspective.aspectRatio);
            znear = static_cast<float>(gcam.perspective.znear);
            if (gcam.perspective.zfar > 0)
                zfar = static_cast<float>(gcam.perspective.zfar);

            add_camera(std::make_shared<Camera>(eye, center, up_vec, fov, aspect, znear, zfar));
            SPDLOG_DEBUG("GLTFScene: loaded {} camera '{}' at ({},{},{})", gcam.type, gcam.name,
                         eye.x, eye.y, eye.z);
        } else {
            SPDLOG_WARN("GLTFScene: Camera with type {} not supported.", gcam.type);
            continue;
        }
    }

    // Fallback: add a default camera if the scene doesn't have one
    if (get_cameras().empty()) {
        SPDLOG_INFO("GLTFScene: no cameras in file, adding default camera");

        add_camera(std::make_shared<Camera>(float3(3, 3, 3), float3(0, 0, 0), get_up(), 60.f,
                                            1920.f / 1080.f, 0.01f, 1000.f));

        if (aabb.is_valid()) {
            get_active_camera()->look_at(float3(1.3) * aabb.get_max().y, aabb.get_center(),
                                         get_up());
            get_active_camera()->look_at_bounding_box(aabb);
        }
    }
}

void GLTFScene::compute_aabb() {
    aabb.reset();

    for (uint32_t gltf_node_index = 0; gltf_node_index < node_map.size(); gltf_node_index++) {
        NodeID node_id = node_map[gltf_node_index];

        if (node_id == NODE_ID_INVALID) {
            // node not used by the current loaded model
            continue;
        }

        const auto& gnode = model->nodes[gltf_node_index];
        if (gnode.mesh >= 0) {
            const auto& mesh = model->meshes[gnode.mesh];
            for (const auto& prim : mesh.primitives) {
                auto it = prim.attributes.find("POSITION");
                if (it == prim.attributes.end())
                    continue;

                const tinygltf::Accessor& accessor = model->accessors[it->second];

                if (accessor.minValues.size() == 3 && accessor.maxValues.size() == 3) {
                    float4 minv = {
                        static_cast<float>(accessor.minValues[0]),
                        static_cast<float>(accessor.minValues[1]),
                        static_cast<float>(accessor.minValues[2]),
                        1,
                    };

                    float4 maxv = {
                        static_cast<float>(accessor.maxValues[0]),
                        static_cast<float>(accessor.maxValues[1]),
                        static_cast<float>(accessor.maxValues[2]),
                        1,
                    };

                    aabb.expand(mul(get_global_transform(node_id), minv));
                    aabb.expand(mul(get_global_transform(node_id), maxv));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

void GLTFScene::load(const CommandBufferHandle& cmd, const std::filesystem::path& path) {
    model = std::make_unique<tinygltf::Model>();
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    SPDLOG_INFO("GLFWScene: loading {}", path.string());

    bool ok;
    if (path.extension() == ".glb") {
        ok = loader.LoadBinaryFromFile(model.get(), &err, &warn, path.string());
    } else {
        ok = loader.LoadASCIIFromFile(model.get(), &err, &warn, path.string());
    }

    if (!warn.empty()) {
        SPDLOG_WARN("GLTFScene: {}", warn);
    }
    if (!ok) {
        throw merian::SceneError(
            fmt::format("GLTFScene: failed to load '{}': {}", path.string(), err));
    }

    // ----------------

    load_materials(cmd);

    load_meshes();

    // Scene graph
    const int scene_index = model->defaultScene >= 0 ? model->defaultScene : 0;
    assert(scene_index < static_cast<int>(model->scenes.size()));
    node_map.resize(model->nodes.size(), NODE_ID_INVALID);
    for (int root_node : model->scenes[scene_index].nodes) {
        load_node(root_node, NODE_ID_INVALID);
    }

    // AABB and cameras need the scene graph
    compute_aabb();

    load_cameras();

    SPDLOG_INFO("GLTFScene: loaded '{}' nodes: {}, meshes: {}, materials: {}, textures: {}",
                path.filename().string(), get_scene_graph().size(), get_meshes().size(),
                material_map.size(), model->images.size());
}

} // namespace merian
