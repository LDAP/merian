#include "merian-shaders/scene/gltf_scene.hpp"
#include "merian/utils/normal_encoding.hpp"

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

// ---------------------------------------------------------------------------
// Accessor helpers
// ---------------------------------------------------------------------------

namespace {

template <typename T> const T* get_accessor_data(const tinygltf::Model& model, int accessor_index) {
    const auto& accessor = model.accessors[accessor_index];
    const auto& bv = model.bufferViews[accessor.bufferView];
    const auto& buf = model.buffers[bv.buffer];
    return reinterpret_cast<const T*>(buf.data.data() + bv.byteOffset + accessor.byteOffset);
}

int get_accessor_byte_stride(const tinygltf::Model& model, int accessor_index) {
    const auto& accessor = model.accessors[accessor_index];
    const auto& bv = model.bufferViews[accessor.bufferView];
    return accessor.ByteStride(bv);
}

template <typename T> T read_strided(const uint8_t* base, int byte_stride, size_t index) {
    return *reinterpret_cast<const T*>(base + byte_stride * index);
}

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

void GLTFScene::load_materials(const CommandBufferHandle& cmd, const tinygltf::Model& model) {
    auto tex_mgr = get_texture_manager();

    // Load glTF images -> TextureIDs
    std::vector<TextureID> image_to_texture;
    image_to_texture.reserve(model.images.size());
    for (const auto& img : model.images) {
        if (img.image.empty() || img.width <= 0 || img.height <= 0) {
            SPDLOG_WARN("GLTFScene: skipping invalid image '{}'", img.name);
            image_to_texture.push_back(TextureID(-1));
            continue;
        }

        // tinygltf decodes images to RGBA 8-bit by default (component == 4)
        if (img.component == 4 && img.bits == 8) {
            auto tid = tex_mgr->add_texture_from_rgba8(
                cmd, reinterpret_cast<const uint32_t*>(img.image.data()),
                static_cast<uint32_t>(img.width), static_cast<uint32_t>(img.height));
            image_to_texture.push_back(tid);
        } else {
            SPDLOG_WARN("GLTFScene: unsupported image format ({} components, {} bits) for '{}'",
                        img.component, img.bits, img.name);
            image_to_texture.push_back(TextureID(-1));
        }
    }

    // Convert glTF materials -> DiffuseMaterial
    material_map.resize(model.materials.size());
    for (size_t i = 0; i < model.materials.size(); i++) {
        const auto& gmat = model.materials[i];
        const auto& pbr = gmat.pbrMetallicRoughness;

        DiffuseMaterial mat;
        mat.base_color_factor = float4(
            static_cast<float>(pbr.baseColorFactor[0]), static_cast<float>(pbr.baseColorFactor[1]),
            static_cast<float>(pbr.baseColorFactor[2]), static_cast<float>(pbr.baseColorFactor[3]));

        // Alpha texture from base color texture
        if (pbr.baseColorTexture.index >= 0) {
            const auto& tex = model.textures[pbr.baseColorTexture.index];
            if (tex.source >= 0 && tex.source < static_cast<int>(image_to_texture.size())) {
                mat.header.alpha_texture_id = image_to_texture[tex.source];
            }
        }

        material_map[i] = get_material_system()->add_material(diffuse_type_id, mat);
    }

    // Default material for primitives without one
    if (material_map.empty()) {
        DiffuseMaterial default_mat;
        material_map.push_back(get_material_system()->add_material(diffuse_type_id, default_mat));
    }
}

void GLTFScene::load_meshes(const tinygltf::Model& model) {
    mesh_map.resize(model.meshes.size());

    for (uint32_t gltf_mesh_id = 0; gltf_mesh_id < model.meshes.size(); gltf_mesh_id++) {
        const auto& gmesh = model.meshes[gltf_mesh_id];

        for (uint32_t primitive_index = 0; primitive_index < gmesh.primitives.size();
             primitive_index++) {
            const auto& prim = gmesh.primitives[primitive_index];

            if (prim.mode != -1 && prim.mode != TINYGLTF_MODE_TRIANGLES)
                continue;

            auto pos_it = prim.attributes.find("POSITION");
            if (pos_it == prim.attributes.end())
                continue;

            const auto& pos_acc = model.accessors[pos_it->second];
            const size_t vertex_count = pos_acc.count;

            // Get attribute data pointers and strides
            const uint8_t* pos_base =
                reinterpret_cast<const uint8_t*>(get_accessor_data<uint8_t>(model, pos_it->second));
            int pos_stride = get_accessor_byte_stride(model, pos_it->second);

            const uint8_t* nrm_base = nullptr;
            int nrm_stride = 0;
            auto nrm_it = prim.attributes.find("NORMAL");
            if (nrm_it != prim.attributes.end()) {
                nrm_base = reinterpret_cast<const uint8_t*>(
                    get_accessor_data<uint8_t>(model, nrm_it->second));
                nrm_stride = get_accessor_byte_stride(model, nrm_it->second);
            }

            const uint8_t* uv_base = nullptr;
            int uv_stride = 0;
            auto uv_it = prim.attributes.find("TEXCOORD_0");
            if (uv_it != prim.attributes.end()) {
                uv_base = reinterpret_cast<const uint8_t*>(
                    get_accessor_data<uint8_t>(model, uv_it->second));
                uv_stride = get_accessor_byte_stride(model, uv_it->second);
            }

            const uint8_t* tan_base = nullptr;
            int tan_stride = 0;
            auto tan_it = prim.attributes.find("TANGENT");
            if (tan_it != prim.attributes.end()) {
                tan_base = reinterpret_cast<const uint8_t*>(
                    get_accessor_data<uint8_t>(model, tan_it->second));
                tan_stride = get_accessor_byte_stride(model, tan_it->second);
            }

            std::vector<PackedVertexData> vertices(vertex_count);
            for (size_t v = 0; v < vertex_count; v++) {
                auto& vd = vertices[v];

                auto pos = read_strided<float3>(pos_base, pos_stride, v);
                vd.position = pos;

                if (nrm_base) {
                    auto n = read_strided<float3>(nrm_base, nrm_stride, v);
                    vd.encoded_normal = encode_normal(n);
                } else {
                    vd.encoded_normal = encode_normal(float3(0, 1, 0));
                }

                if (uv_base) {
                    auto uv = read_strided<float2>(uv_base, uv_stride, v);
                    vd.uv = half2(uv.x, uv.y);
                } else {
                    vd.uv = half2(0, 0);
                }

                if (tan_base) {
                    auto t = read_strided<float4>(tan_base, tan_stride, v);
                    float3 tangent_dir(t.x, t.y, t.z);
                    uint32_t enc = encode_normal(tangent_dir);
                    // Store sign bit in the LSB
                    enc = (enc & ~1u) | (t.w < 0.f ? 1u : 0u);
                    vd.encoded_tangent = enc;
                } else {
                    vd.encoded_tangent = 0;
                }
            }

            std::vector<uint3> indices;
            if (prim.indices >= 0) {
                const auto& idx_acc = model.accessors[prim.indices];
                const size_t tri_count = idx_acc.count / 3;
                indices.resize(tri_count);

                const auto& bv = model.bufferViews[idx_acc.bufferView];
                const auto& buf = model.buffers[bv.buffer];
                const uint8_t* idx_base = buf.data.data() + bv.byteOffset + idx_acc.byteOffset;

                if (idx_acc.componentType == 5123 /* UNSIGNED_SHORT */) {
                    const auto* src = reinterpret_cast<const uint16_t*>(idx_base);
                    for (size_t t = 0; t < tri_count; t++) {
                        indices[t] = uint3(src[t * 3 + 0], src[t * 3 + 1], src[t * 3 + 2]);
                    }
                } else if (idx_acc.componentType == 5125 /* UNSIGNED_INT */) {
                    const auto* src = reinterpret_cast<const uint32_t*>(idx_base);
                    for (size_t t = 0; t < tri_count; t++) {
                        indices[t] = uint3(src[t * 3 + 0], src[t * 3 + 1], src[t * 3 + 2]);
                    }
                } else if (idx_acc.componentType == 5121 /* UNSIGNED_BYTE */) {
                    for (size_t t = 0; t < tri_count; t++) {
                        indices[t] =
                            uint3(idx_base[t * 3 + 0], idx_base[t * 3 + 1], idx_base[t * 3 + 2]);
                    }
                }
            } else {
                // Non-indexed: generate sequential indices
                const size_t tri_count = vertex_count / 3;
                indices.resize(tri_count);
                for (size_t t = 0; t < tri_count; t++) {
                    indices[t] =
                        uint3(static_cast<uint32_t>(t * 3), static_cast<uint32_t>(t * 3 + 1),
                              static_cast<uint32_t>(t * 3 + 2));
                }
            }

            // Determine material
            MaterialID mat_id = material_map[0]; // default
            if (prim.material >= 0 && prim.material < static_cast<int>(material_map.size())) {
                mat_id = material_map[prim.material];
            }

            // Determine flags
            GeometryFlags flags = GeometryFlags::None;
            if (prim.material >= 0 && prim.material < static_cast<int>(model.materials.size())) {
                if (model.materials[prim.material].alphaMode == "OPAQUE") {
                    flags = flags | GeometryFlags::IsOpaque;
                }
            } else {
                flags = flags | GeometryFlags::IsOpaque;
            }

            Mesh mesh;
            mesh.name =
                gmesh.primitives.size() > 1
                    ? fmt::format("{} ({:02})",
                                  gmesh.name.empty() ? fmt::format("GLTF Mesh {}", gltf_mesh_id)
                                                     : gmesh.name,
                                  primitive_index)
                    : gmesh.name;
            mesh.vertices = std::move(vertices);
            mesh.indices = std::move(indices);
            mesh.material_id = mat_id;
            mesh.flags = flags;

            const MeshID mesh_id = add_mesh(std::move(mesh));
            mesh_map[gltf_mesh_id].emplace_back(mesh_id);
        }
    }
}

void GLTFScene::load_node(const tinygltf::Model& model, int gltf_node_index, NodeID parent_id) {
    const auto& gnode = model.nodes[gltf_node_index];

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
        load_node(model, child_idx, nid);
    }
}

void GLTFScene::load_cameras(const tinygltf::Model& model) {
    for (size_t ni = 0; ni < model.nodes.size(); ni++) {
        const auto& gnode = model.nodes[ni];
        if (gnode.camera < 0 || gnode.camera >= static_cast<int>(model.cameras.size()))
            continue;

        const auto& gcam = model.cameras[gnode.camera];
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

void GLTFScene::compute_aabb(const tinygltf::Model& model) {
    aabb.reset();

    for (uint32_t gltf_node_index = 0; gltf_node_index < node_map.size(); gltf_node_index++) {
        NodeID node_id = node_map[gltf_node_index];

        if (node_id == NODE_ID_INVALID) {
            // node not used by the current loaded model
            continue;
        }

        const auto& gnode = model.nodes[gltf_node_index];
        if (gnode.mesh >= 0) {
            const auto& mesh = model.meshes[gnode.mesh];
            for (const auto& prim : mesh.primitives) {
                auto it = prim.attributes.find("POSITION");
                if (it == prim.attributes.end())
                    continue;

                const tinygltf::Accessor& accessor = model.accessors[it->second];

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
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ok;
    if (path.extension() == ".glb") {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());
    }

    if (!warn.empty()) {
        SPDLOG_WARN("GLTFScene: {}", warn);
    }
    if (!ok) {
        throw merian::SceneError(
            fmt::format("GLTFScene: failed to load '{}': {}", path.string(), err));
    }

    // ----------------

    load_materials(cmd, model);

    load_meshes(model);

    // Scene graph
    const int scene_index = model.defaultScene >= 0 ? model.defaultScene : 0;
    assert(scene_index < static_cast<int>(model.scenes.size()));
    node_map.resize(model.nodes.size(), NODE_ID_INVALID);
    for (int root_node : model.scenes[scene_index].nodes) {
        load_node(model, root_node, NODE_ID_INVALID);
    }

    // AABB and cameras need the scene graph
    compute_aabb(model);

    load_cameras(model);

    SPDLOG_INFO("GLTFScene: loaded '{}' nodes: {}, meshes: {}, materials: {}, textures: {}",
                path.filename().string(), get_scene_graph().size(), get_meshes().size(),
                material_map.size(), model.images.size());
}

} // namespace merian
