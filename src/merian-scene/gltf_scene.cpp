#include "merian-scene/gltf_scene.hpp"

#include "merian-scene/gltf_material.hpp"
#include "merian/utils/normal_encoding.hpp"
#include "merian/vk/utils/math.hpp"

#include <algorithm>
#include <cmath>
#include <fmt/format.h>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <limits>
#include <spdlog/spdlog.h>
#include <tiny_gltf.h>

namespace merian {

GLTFScene::GLTFScene(const ShaderCompileContextHandle& compile_context,
                     const ContextHandle& context,
                     const ResourceAllocatorHandle& allocator,
                     const MaterialSystemHandle& material_system)
    : Scene(compile_context, context, allocator, material_system) {

    gltf_type_id = material_system->register_material_type(GLTF_MATERIAL_SLANG_TYPE_NAME,
                                                           GLTF_MATERIAL_SLANG_MODULE_PATH);
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
//
// glTF meshes carry strided + arbitrarily-typed source data; they aren't in
// the canonical PackedVertexData layout. GLTFMesh implements
// HostVertexSource and HostIndexSource for bulk write into device-local buffers.
class GLTFMesh : public Scene::Mesh,
                 public Scene::Mesh::HostVertexSource<PackedVertexData>,
                 public Scene::Mesh::HostIndexSource {
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

    const uint8_t* idx_base = nullptr; // null -> sequential indices
    uint32_t primitive_count = 0;

    void set_indices(const uint8_t* base, int gltf_component_type, uint32_t prim_count) {
        idx_base = base;
        primitive_count = prim_count;
        if (base == nullptr) {
            // No glTF indices: triangles use sequential vertex layout.
            index_type = vk::IndexType::eNoneKHR;
        } else if (gltf_component_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            index_type = vk::IndexType::eUint16;
        } else if (gltf_component_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            index_type = vk::IndexType::eUint32;
        } else {
            index_type = vk::IndexType::eUint8;
        }
    }

    uint32_t get_vertex_count() const override {
        return vertex_count;
    }
    uint32_t get_primitive_count() const override {
        return primitive_count;
    }

    MeshVertexData get_vertices() const override {
        return static_cast<HostVertices>(const_cast<GLTFMesh*>(this));
    }
    MeshPrevVertexData get_prev_vertices() const override {
        return std::monostate{};
    }
    MeshIndexData get_indices() const override {
        if (idx_base == nullptr)
            return std::monostate{};
        return static_cast<HostIndices>(const_cast<GLTFMesh*>(this));
    }

    // ---- HostVertexSource<PackedVertexData> ----

    void write(PackedVertexData* dst) const override {
        for (uint32_t v = 0; v < vertex_count; v++) {
            const float3 pos = get_position(v);
            const float3 nrm = get_normal(v);
            const float2 uv = get_uv(v);
            const float4 tan = get_tangent(v);
            dst[v].position = pos;
            dst[v].encoded_normal = encode_normal(nrm);
            dst[v].uv = half2(uv.x, uv.y);
            dst[v].encoded_tangent = tan_base ? encode_tangent(tan) : 0u;
        }
    }

    // ---- HostIndexSource ----

    void write(void* dst) const override {
        assert(idx_base != nullptr);
        std::memcpy(dst, idx_base, primitive_count * 3 * size_for_index_type(index_type));
    }

  private:
    float3 get_position(uint32_t v) const {
        return read_strided<float3>(pos_base, pos_stride, v);
    }
    float3 get_normal(uint32_t v) const {
        if (nrm_base == nullptr)
            return float3(0, 1, 0);
        return read_strided<float3>(nrm_base, nrm_stride, v);
    }
    float2 get_uv(uint32_t v) const {
        if (uv_base == nullptr)
            return float2(0, 0);
        return read_strided<float2>(uv_base, uv_stride, v);
    }
    float4 get_tangent(uint32_t v) const {
        if (tan_base == nullptr)
            return float4(1, 0, 0, 1);
        return read_strided<float4>(tan_base, tan_stride, v);
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
    default_gltf_sampler = sampler_pool->for_filter_and_address_mode(
        vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerAddressMode::eRepeat,
        vk::SamplerMipmapMode::eLinear, false);

    // Reset slot table; textures upload lazily on first material access.
    texture_slots.assign(model->textures.size(), GltfTextureSlot{});

    // Build materials. get_or_load_texture pulls in only the textures actually referenced.
    material_map.resize(model->materials.size());
    bool any_transmissive = false;
    bool any_volume = false;
    bool any_clearcoat = false;
    bool any_sheen = false;
    bool any_iridescence = false;
    bool any_anisotropy = false;
    for (size_t i = 0; i < model->materials.size(); i++) {
        const auto& gmat = model->materials[i];
        const auto& pbr = gmat.pbrMetallicRoughness;

        GltfMaterial mat;

        // base color (sRGB texture × linear factor)
        mat.payload.base_color_factor = float4(
            static_cast<float>(pbr.baseColorFactor[0]), static_cast<float>(pbr.baseColorFactor[1]),
            static_cast<float>(pbr.baseColorFactor[2]), static_cast<float>(pbr.baseColorFactor[3]));
        if (pbr.baseColorTexture.index >= 0) {
            mat.header.alpha_texture_id =
                get_or_load_texture(cmd, pbr.baseColorTexture.index, /*linear=*/false);
        }

        // metallic-roughness (linear; spec: B=metallic, G=roughness)
        mat.payload.metallic_factor = static_cast<float>(pbr.metallicFactor);
        mat.payload.roughness_factor = static_cast<float>(pbr.roughnessFactor);
        if (pbr.metallicRoughnessTexture.index >= 0) {
            mat.payload.metallic_roughness_texture =
                get_or_load_texture(cmd, pbr.metallicRoughnessTexture.index, /*linear=*/true);
        }

        // normal map (linear) + scale
        if (gmat.normalTexture.index >= 0) {
            mat.payload.normal_texture =
                get_or_load_texture(cmd, gmat.normalTexture.index, /*linear=*/true);
            mat.payload.normal_scale = static_cast<float>(gmat.normalTexture.scale);
        }

        // emissive (sRGB texture × linear factor)
        mat.payload.emissive_factor = float3(static_cast<float>(gmat.emissiveFactor[0]),
                                             static_cast<float>(gmat.emissiveFactor[1]),
                                             static_cast<float>(gmat.emissiveFactor[2]));
        if (gmat.emissiveTexture.index >= 0) {
            mat.payload.emissive_texture =
                get_or_load_texture(cmd, gmat.emissiveTexture.index, /*linear=*/false);
        }

        // occlusion (linear) + strength
        if (gmat.occlusionTexture.index >= 0) {
            mat.payload.occlusion_texture =
                get_or_load_texture(cmd, gmat.occlusionTexture.index, /*linear=*/true);
            mat.payload.occlusion_strength = static_cast<float>(gmat.occlusionTexture.strength);
        }

        // KHR_materials_transmission / _ior / _volume — refractive glass.
        if (const auto it = gmat.extensions.find("KHR_materials_transmission");
            it != gmat.extensions.end() && it->second.Has("transmissionFactor")) {
            mat.payload.transmission_weight =
                static_cast<float>(it->second.Get("transmissionFactor").GetNumberAsDouble());
            any_transmissive |= mat.payload.transmission_weight > 0.0f;
        }
        if (const auto it = gmat.extensions.find("KHR_materials_ior");
            it != gmat.extensions.end() && it->second.Has("ior")) {
            mat.payload.ior = static_cast<float>(it->second.Get("ior").GetNumberAsDouble());
        }
        // KHR_materials_volume → Beer-Lambert absorption: sigma_a = -ln(attenuationColor) /
        // attenuationDistance. thicknessFactor == 0 means thin-walled (no volume), so it only
        // gates volume vs. thin-walled here. The integrator applies the absorption over the actual
        // world-space distance the ray travels inside the medium, and attenuationDistance is
        // world-space, so the absorption needs no node-scale adjustment (unlike a rasterizer that
        // uses the mesh-local thicknessFactor as the path length).
        if (const auto it = gmat.extensions.find("KHR_materials_volume");
            it != gmat.extensions.end()) {
            const tinygltf::Value& ext = it->second;
            const double thickness_factor =
                ext.Has("thicknessFactor") ? ext.Get("thicknessFactor").GetNumberAsDouble() : 0.0;
            float3 att_color(1.0f, 1.0f, 1.0f);
            if (ext.Has("attenuationColor")) {
                const tinygltf::Value& c = ext.Get("attenuationColor");
                if (c.ArrayLen() >= 3) {
                    att_color = float3(static_cast<float>(c.Get(0).GetNumberAsDouble()),
                                       static_cast<float>(c.Get(1).GetNumberAsDouble()),
                                       static_cast<float>(c.Get(2).GetNumberAsDouble()));
                }
            }
            const double att_dist = ext.Has("attenuationDistance")
                                        ? ext.Get("attenuationDistance").GetNumberAsDouble()
                                        : std::numeric_limits<double>::infinity();
            if (thickness_factor > 0.0 && std::isfinite(att_dist) && att_dist > 0.0) {
                const auto sigma = [att_dist](float c) {
                    return static_cast<float>(-std::log(std::clamp(c, 1e-4f, 1.0f)) / att_dist);
                };
                mat.payload.absorption =
                    float3(sigma(att_color.x), sigma(att_color.y), sigma(att_color.z));
                any_volume = true;
            }
        }

        // KHR_materials_clearcoat — thin glossy dielectric coat (textures store R = factor, G =
        // roughness).
        if (const auto it = gmat.extensions.find("KHR_materials_clearcoat");
            it != gmat.extensions.end()) {
            const tinygltf::Value& ext = it->second;
            if (ext.Has("clearcoatFactor")) {
                mat.payload.clearcoat_weight =
                    static_cast<float>(ext.Get("clearcoatFactor").GetNumberAsDouble());
            }
            if (ext.Has("clearcoatRoughnessFactor")) {
                mat.payload.clearcoat_roughness =
                    static_cast<float>(ext.Get("clearcoatRoughnessFactor").GetNumberAsDouble());
            }
            const auto ext_texture = [&](const char* key) -> int {
                if (ext.Has(key) && ext.Get(key).Has("index")) {
                    return static_cast<int>(ext.Get(key).Get("index").GetNumberAsDouble());
                }
                return -1;
            };
            if (const int idx = ext_texture("clearcoatTexture"); idx >= 0) {
                mat.payload.clearcoat_texture = get_or_load_texture(cmd, idx, /*linear=*/true);
            }
            if (const int idx = ext_texture("clearcoatRoughnessTexture"); idx >= 0) {
                mat.payload.clearcoat_roughness_texture =
                    get_or_load_texture(cmd, idx, /*linear=*/true);
            }
            any_clearcoat |= mat.payload.clearcoat_weight > 0.0f;
        }

        // KHR_materials_sheen — retroreflective microfibre layer (colour in sRGB, roughness in the
        // sheenRoughnessTexture alpha channel).
        if (const auto it = gmat.extensions.find("KHR_materials_sheen");
            it != gmat.extensions.end()) {
            const tinygltf::Value& ext = it->second;
            if (ext.Has("sheenColorFactor")) {
                const tinygltf::Value& c = ext.Get("sheenColorFactor");
                if (c.ArrayLen() >= 3) {
                    mat.payload.sheen_color =
                        float3(static_cast<float>(c.Get(0).GetNumberAsDouble()),
                               static_cast<float>(c.Get(1).GetNumberAsDouble()),
                               static_cast<float>(c.Get(2).GetNumberAsDouble()));
                }
            }
            if (ext.Has("sheenRoughnessFactor")) {
                mat.payload.sheen_roughness =
                    static_cast<float>(ext.Get("sheenRoughnessFactor").GetNumberAsDouble());
            }
            const auto ext_texture = [&](const char* key) -> int {
                if (ext.Has(key) && ext.Get(key).Has("index")) {
                    return static_cast<int>(ext.Get(key).Get("index").GetNumberAsDouble());
                }
                return -1;
            };
            if (const int idx = ext_texture("sheenColorTexture"); idx >= 0) {
                mat.payload.sheen_color_texture = get_or_load_texture(cmd, idx, /*linear=*/false);
            }
            if (const int idx = ext_texture("sheenRoughnessTexture"); idx >= 0) {
                mat.payload.sheen_roughness_texture =
                    get_or_load_texture(cmd, idx, /*linear=*/true);
            }
            any_sheen |= mat.payload.sheen_color.r > 0.0f || mat.payload.sheen_color.g > 0.0f ||
                         mat.payload.sheen_color.b > 0.0f;
        }

        // KHR_materials_iridescence — thin-film interference on the specular Fresnel.
        if (const auto it = gmat.extensions.find("KHR_materials_iridescence");
            it != gmat.extensions.end()) {
            const tinygltf::Value& ext = it->second;
            if (ext.Has("iridescenceFactor")) {
                mat.payload.iridescence_factor =
                    static_cast<float>(ext.Get("iridescenceFactor").GetNumberAsDouble());
            }
            if (ext.Has("iridescenceIor")) {
                mat.payload.iridescence_ior =
                    static_cast<float>(ext.Get("iridescenceIor").GetNumberAsDouble());
            }
            if (ext.Has("iridescenceThicknessMinimum")) {
                mat.payload.iridescence_thickness_min =
                    static_cast<float>(ext.Get("iridescenceThicknessMinimum").GetNumberAsDouble());
            }
            if (ext.Has("iridescenceThicknessMaximum")) {
                mat.payload.iridescence_thickness_max =
                    static_cast<float>(ext.Get("iridescenceThicknessMaximum").GetNumberAsDouble());
            }
            const auto ext_texture = [&](const char* key) -> int {
                if (ext.Has(key) && ext.Get(key).Has("index")) {
                    return static_cast<int>(ext.Get(key).Get("index").GetNumberAsDouble());
                }
                return -1;
            };
            if (const int idx = ext_texture("iridescenceTexture"); idx >= 0) {
                mat.payload.iridescence_texture = get_or_load_texture(cmd, idx, /*linear=*/true);
            }
            if (const int idx = ext_texture("iridescenceThicknessTexture"); idx >= 0) {
                mat.payload.iridescence_thickness_texture =
                    get_or_load_texture(cmd, idx, /*linear=*/true);
            }
            any_iridescence |= mat.payload.iridescence_factor > 0.0f;
        }

        // KHR_materials_anisotropy — direction (texture RG) and strength (texture B) in the
        // tangent plane.
        if (const auto it = gmat.extensions.find("KHR_materials_anisotropy");
            it != gmat.extensions.end()) {
            const tinygltf::Value& ext = it->second;
            if (ext.Has("anisotropyStrength")) {
                mat.payload.anisotropy_strength =
                    static_cast<float>(ext.Get("anisotropyStrength").GetNumberAsDouble());
            }
            if (ext.Has("anisotropyRotation")) {
                mat.payload.anisotropy_rotation =
                    static_cast<float>(ext.Get("anisotropyRotation").GetNumberAsDouble());
            }
            if (ext.Has("anisotropyTexture") && ext.Get("anisotropyTexture").Has("index")) {
                const int idx =
                    static_cast<int>(ext.Get("anisotropyTexture").Get("index").GetNumberAsDouble());
                mat.payload.anisotropy_texture = get_or_load_texture(cmd, idx, /*linear=*/true);
            }
            any_anisotropy |= mat.payload.anisotropy_strength > 0.0f;
        }

        material_map[i] = get_material_system()->add_material(gltf_type_id, mat);
    }

    // Enable exactly the lobes this asset uses: features not needed are disabled so the Slang
    // compiler folds them out of the BRDF. set_enable_* is a no-op when the flag is unchanged.
    get_material_system()->set_enable_transmission(any_transmissive);
    get_material_system()->set_enable_volume(any_volume);
    get_material_system()->set_enable_clearcoat(any_clearcoat);
    get_material_system()->set_enable_sheen(any_sheen);
    get_material_system()->set_enable_iridescence(any_iridescence);
    get_material_system()->set_enable_anisotropy(any_anisotropy);

    // Default material for primitives without one
    if (material_map.empty()) {
        GltfMaterial default_mat;
        material_map.push_back(get_material_system()->add_material(gltf_type_id, default_mat));
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
            SPDLOG_DEBUG("GLTFScene: texture {} ('{}') is sampled as a normal/MR/occlusion map but "
                         "its sampler requests *_MIPMAP_*",
                         gltf_tex_idx, img.name);
        }
        generate_mipmaps = wants_mipmaps;
    } else {
        generate_mipmaps = wants_mipmaps || force_mipmaps_color;
    }

    SPDLOG_DEBUG("GLTFScene: uploading texture {} (image '{}'), linear={}, mips={}", gltf_tex_idx,
                 img.name, linear, generate_mipmaps);

    TextureHandle texture = get_allocator()->create_texture_from_rgba8(
        cmd, reinterpret_cast<const uint32_t*>(img.image.data()), static_cast<uint32_t>(img.width),
        static_cast<uint32_t>(img.height), sampler, !linear, img.name, generate_mipmaps);
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
                mesh->set_indices(get_accessor_base(*model, prim.indices), idx_acc.componentType,
                                  static_cast<uint32_t>(idx_acc.count / 3));
            } else {
                mesh->set_indices(nullptr, 0, mesh->vertex_count / 3);
            }

            // Determine material
            MaterialID mat_id = material_map[0]; // default
            if (prim.material >= 0 && prim.material < static_cast<int>(material_map.size())) {
                mat_id = material_map[prim.material];
            }

            // Determine flags
            MeshFlags flags = MeshFlags::None;
            if (prim.material >= 0 && prim.material < static_cast<int>(model->materials.size())) {
                const auto& gmat = model->materials[prim.material];
                if (gmat.alphaMode == "OPAQUE") {
                    flags = flags | MeshFlags::IsOpaque;
                }
                // Two-sided, or transmissive: refraction must hit the back/exit interface.
                if (gmat.doubleSided ||
                    gmat.extensions.find("KHR_materials_transmission") != gmat.extensions.end()) {
                    flags = flags | MeshFlags::TwoSided;
                }
            } else {
                flags = flags | MeshFlags::IsOpaque;
            }
            if (mesh->tan_base != nullptr) {
                flags = flags | MeshFlags::HasTangents;
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

    Scene::Node sn;
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
            float yfov = radians(60.f);
            float aspect = 1.f;
            float znear = 0.01f;
            float zfar = 1000.f;

            if (gcam.perspective.aspectRatio > 0)
                aspect = static_cast<float>(gcam.perspective.aspectRatio);
            yfov = static_cast<float>(gcam.perspective.yfov);
            znear = static_cast<float>(gcam.perspective.znear);
            if (gcam.perspective.zfar > 0)
                zfar = static_cast<float>(gcam.perspective.zfar);

            add_camera(std::make_shared<Camera>(eye, center, up_vec, yfov, aspect, znear, zfar));
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

        add_camera(std::make_shared<Camera>(float3(3, 3, 3), float3(0, 0, 0), get_up(),
                                            radians(60.f), 1920.f / 1080.f, 0.01f, 1000.f));
        AABB& aabb = get_aabb();
        if (aabb.is_valid()) {
            get_active_camera()->look_at(float3(1.3) * aabb.get_max().y, aabb.get_center(),
                                         get_up());
            get_active_camera()->look_at_bounding_box(aabb);
        }
    }

    for (const CameraHandle& cam : get_cameras())
        cam->set_jitter_sequence(Camera::JitterSequence::R2);
}

void GLTFScene::compute_aabb() {
    AABB& aabb = get_aabb();
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
    // Drop everything we put into the scene / material system / texture manager on the previous
    // load so this can be called repeatedly. Defer texture destruction to pool reset so any
    // in-flight frame keeps its bindings valid.
    for (const auto& slot : texture_slots) {
        if (slot.id_srgb != TextureID(-1)) {
            cmd->keep_until_pool_reset(get_texture_manager()->get_texture(slot.id_srgb));
            get_texture_manager()->remove_texture(slot.id_srgb);
        }
        if (slot.id_linear != TextureID(-1)) {
            cmd->keep_until_pool_reset(get_texture_manager()->get_texture(slot.id_linear));
            get_texture_manager()->remove_texture(slot.id_linear);
        }
    }
    texture_slots.clear();
    material_map.clear();
    node_map.clear();
    mesh_map.clear();
    gltf_samplers.clear();
    gltf_sampler_wants_mipmaps.clear();
    default_gltf_sampler.reset();

    clear_geometry();
    get_material_system()->clear();
    model.reset();

    SPDLOG_INFO("GLFWScene: loading {}", path.string());

    auto parsed = std::make_unique<tinygltf::Model>();
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ok;
    if (path.extension() == ".glb") {
        ok = loader.LoadBinaryFromFile(parsed.get(), &err, &warn, path.string());
    } else {
        ok = loader.LoadASCIIFromFile(parsed.get(), &err, &warn, path.string());
    }

    if (!warn.empty()) {
        SPDLOG_WARN("GLTFScene: {}", warn);
    }
    if (!ok) {
        // Leave model null so is_ready() reports false and update() bails out cleanly.
        SPDLOG_ERROR("GLTFScene: failed to load '{}': {}", path.string(), err);
        return;
    }
    model = std::move(parsed);

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
                path.filename().string(), get_scene_graph().size(), get_mesh_infos().size(),
                material_map.size(), model->images.size());
}

} // namespace merian
