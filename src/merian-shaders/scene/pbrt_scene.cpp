#include "merian-shaders/scene/pbrt_scene.hpp"

#include "merian-shaders/scene/env_map.hpp"
#include "merian-shaders/shading/materials/openpbr_material.hpp"
#include "merian/io/image_io.hpp"
#include "merian/utils/normal_encoding.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include "pbrt/pbrt_gzip.hpp"
#include "pbrt/pbrt_parser.hpp"
#include "pbrt/pbrt_spectrum.hpp"

#include <fmt/format.h>
#include <miniply.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>

namespace merian {

using pbrt::ParamDict;
using pbrt::ParsedParameter;
using pbrt::ShapeDesc;

namespace {

constexpr int MAX_RESOLVE_DEPTH = 8;

float3 transform_point(const float4x4& m, const float3& p) {
    return float3(mul(m, float4(p, 1.f)));
}

float det3(const float4x4& m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

std::optional<float3> param_rgb(const ParsedParameter& param, const std::filesystem::path& base) {
    if (param.type == "rgb" || param.type == "color") {
        if (param.floats.size() >= 3) {
            return float3(param.floats[0], param.floats[1], param.floats[2]);
        }
        return std::nullopt;
    }
    if (param.type == "float" && !param.floats.empty()) {
        return float3(param.floats[0]);
    }
    return pbrt::spectrum_param_rgb(param, base);
}

std::vector<float3> compute_vertex_normals(const std::vector<float3>& positions,
                                           const std::vector<uint3>& triangles) {
    std::vector<float3> normals(positions.size(), float3(0));
    for (const uint3& tri : triangles) {
        // unnormalized cross = area weighting
        const float3 face =
            cross(positions[tri.y] - positions[tri.x], positions[tri.z] - positions[tri.x]);
        normals[tri.x] += face;
        normals[tri.y] += face;
        normals[tri.z] += face;
    }
    for (float3& n : normals) {
        n = length(n) > 1e-20f ? normalize(n) : float3(0, 1, 0);
    }
    return normals;
}

void pack_vertices(Scene::SimpleMesh& sm,
                   const std::vector<float3>& positions,
                   const std::vector<float3>& normals,
                   const std::vector<float2>& uvs,
                   const std::vector<float3>& tangents) {
    sm.vertices.resize(positions.size());
    for (size_t i = 0; i < positions.size(); i++) {
        PackedVertexData& v = sm.vertices[i];
        v.position = positions[i];
        v.encoded_normal = encode_normal(normals[i]);
        // pbrt uv origin is bottom-left; Vulkan samples top-left.
        v.uv = i < uvs.size() ? half2(uvs[i].x, 1.f - uvs[i].y) : half2(0.f, 0.f);
        v.encoded_tangent = i < tangents.size() ? encode_tangent(float4(tangents[i], 1.f)) : 0u;
    }
}

} // namespace

struct PBRTScene::MaterialBuild {
    OpenPBRMaterial material;
    MeshFlags flags = MeshFlags::IsOpaque;
};

PBRTScene::PBRTScene(const ShaderCompileContextHandle& compile_context,
                     const ContextHandle& context,
                     const ResourceAllocatorHandle& allocator,
                     const MaterialSystemHandle& material_system)
    : Scene(compile_context, context, allocator, material_system) {}

PBRTScene::~PBRTScene() = default;

void PBRTScene::warn_once(const std::string& key, const std::string& message) {
    if (warned.insert(key).second) {
        SPDLOG_WARN("PBRTScene: {}", message);
    }
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

TextureID PBRTScene::load_image_texture(const CommandBufferHandle& cmd,
                                        const std::string& filename,
                                        const bool srgb,
                                        bool* out_has_alpha) {
    TextureSlot& slot = texture_slots[filename];
    TextureID& cached = srgb ? slot.id_srgb : slot.id_linear;
    if (cached != TextureID(-1)) {
        if (out_has_alpha != nullptr) {
            *out_has_alpha = slot.has_alpha;
        }
        return cached;
    }

    const std::filesystem::path path = std::filesystem::path(filename).is_absolute()
                                           ? std::filesystem::path(filename)
                                           : base_dir / filename;
    if (!std::filesystem::exists(path)) {
        SPDLOG_WARN("PBRTScene: texture file not found: {}", path.string());
        return TextureID(-1);
    }

    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](const unsigned char c) { return std::tolower(c); });
    if (ext == ".exr") {
        warn_once("exr", "EXR images are unsupported, skipping " + filename);
        return TextureID(-1);
    }

    // Mipmap color textures only; box-filtering scalar/normal maps degrades them.
    TextureHandle texture;
    try {
        if (ext == ".pfm" || ext == ".hdr") {
            ImageInfo info;
            const BlobHandle blob = image_load_f32(path, info, 4);
            texture = get_allocator()->create_texture_from_rgba32f(
                cmd, blob->get_data<float>(), static_cast<uint32_t>(info.width),
                static_cast<uint32_t>(info.height), vk::SamplerAddressMode::eRepeat,
                vk::Filter::eLinear, vk::Filter::eLinear, path.filename().string(), srgb);
            cmd->barrier(texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
        } else {
            bool has_alpha = false;
            texture = get_allocator()->create_texture_from_file(
                cmd, path, srgb, vk::SamplerAddressMode::eRepeat, vk::Filter::eLinear,
                vk::Filter::eLinear, path.filename().string(), srgb, &has_alpha);
            slot.has_alpha = has_alpha;
        }
    } catch (const std::exception& e) {
        SPDLOG_WARN("PBRTScene: failed to load texture '{}': {}", path.string(), e.what());
        return TextureID(-1);
    }

    if (out_has_alpha != nullptr) {
        *out_has_alpha = slot.has_alpha;
    }
    cached = get_texture_manager()->add_texture(texture);
    return cached;
}

PBRTScene::Resolved PBRTScene::resolve_texture_input(const CommandBufferHandle& cmd,
                                                     const ParamDict& params,
                                                     const char* name,
                                                     const float3& fallback,
                                                     const bool srgb,
                                                     const int depth) {
    const ParsedParameter* param = params.find(name);
    if (param == nullptr) {
        return Resolved{fallback, TextureID(-1), false};
    }
    if (param->type == "texture" && !param->strings.empty()) {
        return resolve_texture_ref(cmd, param->strings[0], srgb, depth + 1);
    }
    return Resolved{param_rgb(*param, base_dir).value_or(fallback), TextureID(-1), false};
}

PBRTScene::Resolved PBRTScene::resolve_texture_ref(const CommandBufferHandle& cmd,
                                                   const std::string& name,
                                                   const bool srgb,
                                                   const int depth) {
    if (depth > MAX_RESOLVE_DEPTH) {
        warn_once("tex_depth", "texture graph too deep, using grey");
        return Resolved{float3(0.5f), TextureID(-1), false};
    }
    const std::string cache_key = name + (srgb ? "|s" : "|l");
    if (const auto it = resolved_textures.find(cache_key); it != resolved_textures.end()) {
        return it->second;
    }

    const pbrt::TextureDesc* tex = desc->find_texture(name);
    if (tex == nullptr) {
        SPDLOG_WARN("PBRTScene: unknown texture '{}'", name);
        return Resolved{float3(0.5f), TextureID(-1), false};
    }
    const ParamDict& p = tex->params;

    Resolved result;
    if (tex->cls == "constant") {
        const ParsedParameter* value = p.find("value");
        result.factor =
            value != nullptr ? param_rgb(*value, base_dir).value_or(float3(1)) : float3(1);
    } else if (tex->cls == "scale") {
        result = resolve_texture_input(cmd, p, "tex", float3(1), srgb, depth);
        const ParsedParameter* s = p.find("scale");
        if (s != nullptr && s->type == "texture") {
            warn_once("tex_scale", "textured 'scale' inputs are unsupported");
        } else if (s != nullptr) {
            result.factor *= param_rgb(*s, base_dir).value_or(float3(1));
        }
    } else if (tex->cls == "imagemap") {
        const std::string filename = p.get_string("filename", "");
        const std::string encoding = p.get_string("encoding", srgb ? "sRGB" : "linear");
        const bool use_srgb = encoding.starts_with("sRGB");
        result.texture = load_image_texture(cmd, filename, use_srgb, &result.has_alpha);
        result.factor = float3(p.get_float("scale", 1.f));
        if (p.get_float("uscale", 1.f) != 1.f || p.get_float("vscale", 1.f) != 1.f ||
            p.get_float("udelta", 0.f) != 0.f || p.get_float("vdelta", 0.f) != 0.f) {
            warn_once("tex_uv", "imagemap uv transforms are unsupported");
        }
        if (p.get_bool("invert", false)) {
            warn_once("tex_invert", "imagemap 'invert' is unsupported");
        }
    } else if (tex->cls == "checkerboard") {
        const Resolved t1 = resolve_texture_input(cmd, p, "tex1", float3(1), srgb, depth);
        const Resolved t2 = resolve_texture_input(cmd, p, "tex2", float3(0), srgb, depth);
        if (t1.texture != TextureID(-1) || t2.texture != TextureID(-1)) {
            warn_once("checker_tex", "textured checkerboard inputs use their constant factors");
        }
        if (p.get_int("dimension", 2) != 2) {
            warn_once("checker_dim", "only 2d checkerboards are supported");
        }

        // Bake the full checker tile so uscale/vscale are exact without per-texture uv
        // transforms. One checker cell per uscale/vscale unit (pbrt: floor(u*uscale)).
        const int32_t cells_x =
            std::clamp(static_cast<int32_t>(std::lround(p.get_float("uscale", 1.f))), 1, 512);
        const int32_t cells_y =
            std::clamp(static_cast<int32_t>(std::lround(p.get_float("vscale", 1.f))), 1, 512);
        const int32_t px = std::clamp(512 / std::max(cells_x, cells_y), 2, 16);
        const int32_t width = cells_x * px;
        const int32_t height = cells_y * px;
        std::vector<float> pixels(static_cast<size_t>(width) * height * 4);
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                const bool odd = ((x / px) + (y / px)) % 2 != 0;
                const float3 c = odd ? t2.factor : t1.factor;
                float* dst = &pixels[(static_cast<size_t>(y) * width + x) * 4];
                dst[0] = c.x;
                dst[1] = c.y;
                dst[2] = c.z;
                dst[3] = 1.f;
            }
        }
        const TextureHandle texture = get_allocator()->create_texture_from_rgba32f(
            cmd, pixels.data(), static_cast<uint32_t>(width), static_cast<uint32_t>(height),
            vk::SamplerAddressMode::eRepeat, vk::Filter::eLinear, vk::Filter::eLinear,
            "pbrt_checker_" + name, true);
        cmd->barrier(texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
        result.texture = get_texture_manager()->add_texture(texture);
    } else if (tex->cls == "mix") {
        const Resolved t1 = resolve_texture_input(cmd, p, "tex1", float3(0), srgb, depth);
        const Resolved t2 = resolve_texture_input(cmd, p, "tex2", float3(1), srgb, depth);
        const float amount = p.get_float("amount", 0.5f);
        if (t1.texture != TextureID(-1) || t2.texture != TextureID(-1)) {
            warn_once("mix_tex", "textured 'mix' inputs are unsupported, using the first input");
            result = t1;
        } else {
            result.factor = lerp(t1.factor, t2.factor, amount);
        }
    } else {
        warn_once("tex_" + tex->cls,
                  fmt::format("texture class '{}' is unsupported, using grey", tex->cls));
        result.factor = float3(0.5f);
    }

    resolved_textures.emplace(cache_key, result);
    return result;
}

PBRTScene::Resolved PBRTScene::resolve_color_param(const CommandBufferHandle& cmd,
                                                   const ParamDict& params,
                                                   const char* name,
                                                   const float3& fallback,
                                                   const bool srgb) {
    const ParsedParameter* param = params.find(name);
    if (param == nullptr) {
        return Resolved{fallback, TextureID(-1), false};
    }
    if (param->type == "texture" && !param->strings.empty()) {
        return resolve_texture_ref(cmd, param->strings[0], srgb, 0);
    }
    return Resolved{param_rgb(*param, base_dir).value_or(fallback), TextureID(-1), false};
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

void PBRTScene::apply_roughness(const CommandBufferHandle& cmd,
                                const ParamDict& params,
                                const std::string& prefix,
                                float& out_roughness,
                                TextureID& out_texture) {
    const ParsedParameter* rough = params.find(prefix + "roughness");
    if (rough != nullptr && rough->type == "texture" && !rough->strings.empty()) {
        const Resolved r = resolve_texture_ref(cmd, rough->strings[0], false, 0);
        out_texture = r.texture;
        out_roughness = r.factor.x;
        return;
    }

    float value = rough != nullptr && !rough->floats.empty() ? rough->floats[0] : 0.f;
    const ParsedParameter* u = params.find(prefix + "uroughness");
    const ParsedParameter* v = params.find(prefix + "vroughness");
    if (u != nullptr || v != nullptr) {
        const float ur = u != nullptr && !u->floats.empty() ? u->floats[0] : value;
        const float vr = v != nullptr && !v->floats.empty() ? v->floats[0] : value;
        value = 0.5f * (ur + vr);
        warn_once("aniso", "anisotropic roughness is averaged to isotropic");
    }
    // remaproughness=true (default): the value is perceptual, matching merian's roughness.
    // Otherwise it is GGX alpha; merian applies alpha = roughness^2.
    if (!params.get_bool("remaproughness", true)) {
        value = std::sqrt(value);
    }
    out_roughness = value;
}

PBRTScene::MaterialBuild PBRTScene::convert_material(const CommandBufferHandle& cmd,
                                                     const int32_t material_index,
                                                     const int depth) {
    MaterialBuild out;
    OpenPBRMaterial& mat = out.material;

    if (material_index < 0 || depth > 4) {
        // pbrt's default material is coateddiffuse with reflectance 0.5.
        mat.base_color = float3(0.5f);
        mat.roughness = 0.f;
        return out;
    }

    const pbrt::MaterialDesc& m = desc->materials[material_index];
    const ParamDict& p = m.params;
    const std::string& type = m.type;

    if (type == "diffuse" || type == "diffusetransmission" || type == "subsurface" ||
        type == "hair" || type == "measured") {
        const Resolved refl = resolve_color_param(cmd, p, "reflectance", float3(0.5f), true);
        mat.base_color = refl.factor;
        mat.header.alpha_texture_id = refl.texture;
        mat.specular_weight = 0.f;
        mat.roughness = 1.f;
        if (refl.has_alpha) {
            out.flags = MeshFlags::TwoSided;
        }
        if (type != "diffuse") {
            warn_once("mat_" + type, fmt::format("material '{}' approximated as diffuse", type));
        }
    } else if (type == "coateddiffuse") {
        const Resolved refl = resolve_color_param(cmd, p, "reflectance", float3(0.5f), true);
        mat.base_color = refl.factor;
        mat.header.alpha_texture_id = refl.texture;
        mat.specular_ior = p.get_float("eta", 1.5f);
        apply_roughness(cmd, p, "", mat.roughness, mat.roughness_texture);
        if (refl.has_alpha) {
            out.flags = MeshFlags::TwoSided;
        }
    } else if (type == "conductor" || type == "coatedconductor") {
        const std::string prefix = type == "coatedconductor" ? "conductor." : "";
        mat.metalness = 1.f;
        if (p.find("reflectance") != nullptr) {
            const Resolved refl = resolve_color_param(cmd, p, "reflectance", float3(0.9f), true);
            mat.base_color = refl.factor;
            mat.header.alpha_texture_id = refl.texture;
        } else {
            const ParsedParameter* eta = p.find(prefix + "eta");
            const ParsedParameter* k = p.find(prefix + "k");
            std::optional<float3> f0;
            if (eta != nullptr && !eta->strings.empty()) {
                f0 = pbrt::named_metal_f0(eta->strings[0]);
            }
            if (!f0 && (eta != nullptr || k != nullptr)) {
                float3 eta_rgb(0.2f);
                float3 k_rgb(3.9f);
                if (eta != nullptr) {
                    eta_rgb = pbrt::spectrum_param_rgb(*eta, base_dir).value_or(eta_rgb);
                }
                if (k != nullptr) {
                    k_rgb = pbrt::spectrum_param_rgb(*k, base_dir).value_or(k_rgb);
                }
                const float3 num = (eta_rgb - 1.f) * (eta_rgb - 1.f) + k_rgb * k_rgb;
                const float3 den = (eta_rgb + 1.f) * (eta_rgb + 1.f) + k_rgb * k_rgb;
                f0 = num / den;
            }
            // pbrt's default conductor is copper.
            mat.base_color = f0.value_or(float3(0.955f, 0.637f, 0.538f));
        }
        apply_roughness(cmd, p, prefix, mat.roughness, mat.roughness_texture);
        if (type == "coatedconductor") {
            warn_once("mat_coatedconductor", "coatedconductor interface layer is ignored");
        }
    } else if (type == "dielectric" || type == "thindielectric") {
        float eta = 1.5f;
        if (const ParsedParameter* e = p.find("eta"); e != nullptr) {
            if (!e->floats.empty()) {
                eta = e->floats[0];
            } else if (!e->strings.empty()) {
                if (const auto named = pbrt::named_glass_ior(e->strings[0])) {
                    eta = *named;
                } else if (const auto rgb = pbrt::spectrum_param_rgb(*e, base_dir)) {
                    eta = (rgb->x + rgb->y + rgb->z) / 3.f;
                    warn_once("dispersive", "spectral dielectric eta is averaged");
                }
            }
        }
        mat.specular_ior = eta;
        mat.transmission = OpenPBRTransmissionData{1.f, float3(1)};
        apply_roughness(cmd, p, "", mat.roughness, mat.roughness_texture);
        out.flags = MeshFlags::TwoSided;
        if (type == "thindielectric") {
            warn_once("mat_thin", "thindielectric approximated as solid dielectric");
        }
    } else if (type == "mix") {
        const ParsedParameter* materials = p.find("materials");
        if (materials != nullptr && materials->strings.size() == 2) {
            // pbrt-v4 MixMaterial: amount is the probability of the second material.
            const float amount = p.get_float("amount", 0.5f);
            const std::string& chosen = materials->strings[amount >= 0.5f ? 1 : 0];
            warn_once("mix_" + m.name + chosen,
                      fmt::format("mix material resolved to '{}'", chosen));
            const auto it = desc->named_material_index.find(chosen);
            if (it != desc->named_material_index.end()) {
                return convert_material(cmd, it->second, depth + 1);
            }
            SPDLOG_WARN("PBRTScene: mix references unknown material '{}'", chosen);
        }
        mat.base_color = float3(0.5f);
        mat.specular_weight = 0.f;
        mat.roughness = 1.f;
    } else {
        warn_once("mat_" + type, fmt::format("material '{}' is unsupported, using diffuse", type));
        mat.base_color = float3(0.5f);
        mat.specular_weight = 0.f;
        mat.roughness = 1.f;
    }

    if (const std::string normal_map = p.get_string("normalmap", ""); !normal_map.empty()) {
        mat.normal_texture = load_image_texture(cmd, normal_map, false, nullptr);
    }
    if (p.find("displacement") != nullptr) {
        warn_once("displacement", "displacement is unsupported");
    }
    return out;
}

MaterialID PBRTScene::material_for_shape(const CommandBufferHandle& cmd,
                                         const ShapeDesc& shape,
                                         MeshFlags& out_flags) {
    const float3 emission = shape.area_light ? shape.area_light->radiance : float3(0);
    const bool twosided = shape.area_light && shape.area_light->twosided;
    const auto key = std::make_tuple(shape.material, emission.x, emission.y, emission.z, twosided,
                                     shape.inside_medium);
    if (const auto it = material_cache.find(key); it != material_cache.end()) {
        out_flags = it->second.flags;
        return it->second.id;
    }

    MaterialBuild build = convert_material(cmd, shape.material, 0);
    build.material.emission = emission;
    if (twosided) {
        build.flags = build.flags | MeshFlags::TwoSided;
    }
    if (shape.inside_medium >= 0) {
        const float3 sigma_a = desc->media[shape.inside_medium].sigma_a;
        if (sigma_a.x > 0.f || sigma_a.y > 0.f || sigma_a.z > 0.f) {
            build.material.volume = OpenPBRVolumeData{sigma_a};
        }
    }

    const auto type_id = get_material_system()->register_material_type(
        build.material.variant_type_name(), OPENPBR_MATERIAL_SLANG_MODULE_PATH);
    const MaterialID id = get_material_system()->add_material(type_id, build.material);

    material_cache.emplace(key, CachedMaterial{id, build.flags});
    out_flags = build.flags;
    return id;
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

namespace {

bool fill_trianglemesh(const ParamDict& p, Scene::SimpleMesh& sm) {
    const std::vector<float3> positions = p.get_vec3_list("P");
    if (positions.empty()) {
        return false;
    }

    std::vector<uint3> triangles;
    if (const std::vector<int32_t>* indices = p.get_int_list("indices"); indices != nullptr) {
        triangles.reserve(indices->size() / 3);
        for (size_t i = 0; i + 2 < indices->size(); i += 3) {
            const uint3 tri(static_cast<uint32_t>((*indices)[i]),
                            static_cast<uint32_t>((*indices)[i + 1]),
                            static_cast<uint32_t>((*indices)[i + 2]));
            if (tri.x >= positions.size() || tri.y >= positions.size() ||
                tri.z >= positions.size()) {
                continue;
            }
            triangles.push_back(tri);
        }
    } else if (positions.size() == 3) {
        triangles.emplace_back(0, 1, 2);
    } else {
        return false;
    }

    std::vector<float3> normals = p.get_vec3_list("N");
    if (normals.size() != positions.size()) {
        normals = compute_vertex_normals(positions, triangles);
    } else {
        for (float3& n : normals) {
            n = length(n) > 1e-20f ? normalize(n) : float3(0, 1, 0);
        }
    }
    std::vector<float2> uvs = p.get_vec2_list("uv");
    if (uvs.empty()) {
        uvs = p.get_vec2_list("st");
    }
    const std::vector<float3> tangents = p.get_vec3_list("S");

    pack_vertices(sm, positions, normals, uvs, tangents);
    sm.indices = std::move(triangles);
    return true;
}

bool fill_plymesh(const std::filesystem::path& path, Scene::SimpleMesh& sm) {
    // miniply reads from a file path; decompress .ply.gz to runtime scratch.
    std::filesystem::path actual = path;
    std::optional<std::filesystem::path> temp;
    if (path.extension() == ".gz") {
        const std::string data = pbrt::gunzip_file(path);
        temp = std::filesystem::temp_directory_path() /
               fmt::format("merian-pbrt-{:x}.ply", std::hash<std::string>{}(path.string()));
        std::ofstream out(*temp, std::ios::binary);
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
        if (!out) {
            return false;
        }
        out.close();
        actual = *temp;
    }

    bool ok = false;
    {
        miniply::PLYReader reader(actual.string().c_str());
        std::vector<float> pos;
        std::vector<float> nrm;
        std::vector<float> uv;
        std::vector<int32_t> tri;
        uint32_t vertex_count = 0;
        bool got_verts = false;
        bool got_faces = false;

        while (reader.valid() && reader.has_element() && (!got_verts || !got_faces)) {
            if (reader.element_is(miniply::kPLYVertexElement)) {
                uint32_t pidx[3];
                if (!reader.load_element() || !reader.find_pos(pidx)) {
                    break;
                }
                vertex_count = reader.num_rows();
                pos.resize(static_cast<size_t>(vertex_count) * 3);
                reader.extract_properties(pidx, 3, miniply::PLYPropertyType::Float, pos.data());
                uint32_t nidx[3];
                if (reader.find_normal(nidx)) {
                    nrm.resize(static_cast<size_t>(vertex_count) * 3);
                    reader.extract_properties(nidx, 3, miniply::PLYPropertyType::Float, nrm.data());
                }
                uint32_t tidx[2];
                if (reader.find_texcoord(tidx)) {
                    uv.resize(static_cast<size_t>(vertex_count) * 2);
                    reader.extract_properties(tidx, 2, miniply::PLYPropertyType::Float, uv.data());
                }
                got_verts = true;
            } else if (reader.element_is(miniply::kPLYFaceElement)) {
                if (!got_verts) {
                    SPDLOG_WARN("PBRTScene: PLY face element before vertex element: {}",
                                path.string());
                    break;
                }
                if (!reader.load_element()) {
                    break;
                }
                uint32_t iidx = reader.find_property("vertex_indices");
                if (iidx == miniply::kInvalidIndex) {
                    iidx = reader.find_property("vertex_index");
                }
                if (iidx == miniply::kInvalidIndex) {
                    break;
                }
                const uint32_t triangle_count = reader.num_triangles(iidx);
                tri.resize(static_cast<size_t>(triangle_count) * 3);
                reader.extract_triangles(iidx, pos.data(), vertex_count,
                                         miniply::PLYPropertyType::Int, tri.data());
                got_faces = true;
            }
            reader.next_element();
        }

        if (got_verts && got_faces) {
            std::vector<float3> positions(vertex_count);
            for (uint32_t i = 0; i < vertex_count; i++) {
                positions[i] = float3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
            }
            std::vector<uint3> triangles;
            triangles.reserve(tri.size() / 3);
            for (size_t i = 0; i + 2 < tri.size(); i += 3) {
                triangles.emplace_back(static_cast<uint32_t>(tri[i]),
                                       static_cast<uint32_t>(tri[i + 1]),
                                       static_cast<uint32_t>(tri[i + 2]));
            }
            std::vector<float3> normals;
            if (nrm.size() == pos.size()) {
                normals.resize(vertex_count);
                for (uint32_t i = 0; i < vertex_count; i++) {
                    const float3 n(nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]);
                    normals[i] = length(n) > 1e-20f ? normalize(n) : float3(0, 1, 0);
                }
            } else {
                normals = compute_vertex_normals(positions, triangles);
            }
            std::vector<float2> uvs;
            if (uv.size() == static_cast<size_t>(vertex_count) * 2) {
                uvs.resize(vertex_count);
                for (uint32_t i = 0; i < vertex_count; i++) {
                    uvs[i] = float2(uv[i * 2], uv[i * 2 + 1]);
                }
            }
            pack_vertices(sm, positions, normals, uvs, {});
            sm.indices = std::move(triangles);
            ok = true;
        }
    }

    if (temp) {
        std::error_code ec;
        std::filesystem::remove(*temp, ec);
    }
    return ok;
}

void fill_sphere(const ParamDict& p, Scene::SimpleMesh& sm) {
    const float radius = p.get_float("radius", 1.f);
    const float zmin = std::clamp(p.get_float("zmin", -radius), -radius, radius);
    const float zmax = std::clamp(p.get_float("zmax", radius), -radius, radius);
    const float phi_max = radians(std::clamp(p.get_float("phimax", 360.f), 0.f, 360.f));
    const float theta_min = std::acos(std::clamp(zmax / radius, -1.f, 1.f));
    const float theta_max = std::acos(std::clamp(zmin / radius, -1.f, 1.f));

    constexpr int32_t SLICES = 64;
    constexpr int32_t STACKS = 32;

    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    positions.reserve(static_cast<size_t>(SLICES + 1) * (STACKS + 1));
    for (int32_t j = 0; j <= STACKS; j++) {
        const float theta = theta_min + (theta_max - theta_min) * static_cast<float>(j) / STACKS;
        for (int32_t i = 0; i <= SLICES; i++) {
            const float phi = phi_max * static_cast<float>(i) / SLICES;
            const float3 n(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
                           std::cos(theta));
            positions.push_back(radius * n);
            normals.push_back(n);
            uvs.emplace_back(static_cast<float>(i) / SLICES, 1.f - static_cast<float>(j) / STACKS);
        }
    }
    std::vector<uint3> triangles;
    triangles.reserve(static_cast<size_t>(SLICES) * STACKS * 2);
    for (int32_t j = 0; j < STACKS; j++) {
        for (int32_t i = 0; i < SLICES; i++) {
            const uint32_t a = j * (SLICES + 1) + i;
            const uint32_t b = a + SLICES + 1;
            triangles.emplace_back(a, b, a + 1);
            triangles.emplace_back(a + 1, b, b + 1);
        }
    }
    pack_vertices(sm, positions, normals, uvs, {});
    sm.indices = std::move(triangles);
}

void fill_disk(const ParamDict& p, Scene::SimpleMesh& sm) {
    const float height = p.get_float("height", 0.f);
    const float radius = p.get_float("radius", 1.f);
    const float inner = std::clamp(p.get_float("innerradius", 0.f), 0.f, radius);
    const float phi_max = radians(std::clamp(p.get_float("phimax", 360.f), 0.f, 360.f));

    constexpr int32_t SLICES = 64;

    std::vector<float3> positions;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    for (int32_t i = 0; i <= SLICES; i++) {
        const float phi = phi_max * static_cast<float>(i) / SLICES;
        const float2 dir(std::cos(phi), std::sin(phi));
        positions.emplace_back(radius * dir.x, radius * dir.y, height);
        positions.emplace_back(inner * dir.x, inner * dir.y, height);
        normals.emplace_back(0, 0, 1);
        normals.emplace_back(0, 0, 1);
        uvs.emplace_back(static_cast<float>(i) / SLICES, 0.f);
        uvs.emplace_back(static_cast<float>(i) / SLICES, 1.f);
    }
    std::vector<uint3> triangles;
    for (int32_t i = 0; i < SLICES; i++) {
        const uint32_t a = i * 2;
        triangles.emplace_back(a, a + 2, a + 1);
        triangles.emplace_back(a + 1, a + 2, a + 3);
    }
    pack_vertices(sm, positions, normals, uvs, {});
    sm.indices = std::move(triangles);
}

} // namespace

std::optional<Scene::MeshID> PBRTScene::build_shape_mesh(const CommandBufferHandle& cmd,
                                                         const size_t shape_index) {
    const ShapeDesc& shape = desc->shapes[shape_index];

    if (shape.material == ShapeDesc::MATERIAL_INTERFACE) {
        SPDLOG_DEBUG("PBRTScene: skipping interface shape {}", shape_index);
        return std::nullopt;
    }
    if (const ParsedParameter* alpha = shape.params.find("alpha"); alpha != nullptr) {
        if (alpha->type == "texture") {
            warn_once("shape_alpha", "textured shape alpha is unsupported");
        } else if (!alpha->floats.empty() && alpha->floats[0] == 0.f) {
            return std::nullopt;
        }
    }

    auto sm = std::make_unique<SimpleMesh>();
    if (shape.type == "trianglemesh" || shape.type == "loopsubdiv") {
        if (!fill_trianglemesh(shape.params, *sm)) {
            SPDLOG_WARN("PBRTScene: invalid trianglemesh (shape {})", shape_index);
            return std::nullopt;
        }
        if (shape.type == "loopsubdiv") {
            warn_once("loopsubdiv", "loading subdivision control meshes unsubdivided");
        }
    } else if (shape.type == "plymesh") {
        const std::string filename = shape.params.get_string("filename", "");
        const std::filesystem::path path = std::filesystem::path(filename).is_absolute()
                                               ? std::filesystem::path(filename)
                                               : base_dir / filename;
        if (filename.empty() || !fill_plymesh(path, *sm)) {
            SPDLOG_WARN("PBRTScene: failed to load plymesh '{}'", filename);
            return std::nullopt;
        }
    } else if (shape.type == "sphere") {
        fill_sphere(shape.params, *sm);
    } else if (shape.type == "disk") {
        fill_disk(shape.params, *sm);
    } else {
        warn_once("shape_" + shape.type,
                  fmt::format("shape '{}' is unsupported, skipping", shape.type));
        return std::nullopt;
    }

    MeshFlags flags = MeshFlags::IsOpaque;
    sm->material_id = material_for_shape(cmd, shape, flags);
    // pbrt geometry is never backface-culled.
    flags = flags | MeshFlags::TwoSided;

    if (shape.reverse_orientation != (det3(shape.ctm) < 0.f)) {
        flags = flags | MeshFlags::FlipFacing;
    }
    if (shape.params.find("S") != nullptr) {
        flags = flags | MeshFlags::HasTangents;
    }
    sm->flags = flags;

    const std::string& material_name =
        shape.material >= 0 ? desc->materials[shape.material].name : "";
    sm->name = material_name.empty() ? fmt::format("{} {:03}", shape.type, shape_index)
                                     : fmt::format("{} ({})", material_name, shape_index);

    AABB& local = shape_aabbs[shape_index];
    for (const PackedVertexData& v : sm->vertices) {
        local.expand(v.position);
    }

    return add_mesh(std::move(sm));
}

// ---------------------------------------------------------------------------
// Scene graph
// ---------------------------------------------------------------------------

void PBRTScene::build_scene(const CommandBufferHandle& cmd) {
    Node root;
    root.name = "pbrt";
    const NodeID root_id = add_node(root);

    std::vector<uint32_t> object_use(desc->objects.size(), 0);
    for (const pbrt::InstanceDesc& instance : desc->instances) {
        object_use[instance.object]++;
    }

    shape_meshes.assign(desc->shapes.size(), std::nullopt);
    shape_aabbs.assign(desc->shapes.size(), AABB{});

    for (size_t i = 0; i < desc->shapes.size(); i++) {
        const ShapeDesc& shape = desc->shapes[i];
        if (shape.object >= 0 && object_use[shape.object] == 0) {
            continue;
        }
        shape_meshes[i] = build_shape_mesh(cmd, i);
    }

    AABB& aabb = get_aabb();
    aabb.reset();
    const auto expand_by = [&](const AABB& local, const float4x4& transform) {
        if (!local.is_valid()) {
            return;
        }
        for (uint32_t c = 0; c < 8; c++) {
            aabb.expand(transform_point(transform, local.get_corner(c)));
        }
    };

    // 1. directly placed shapes
    for (size_t i = 0; i < desc->shapes.size(); i++) {
        const ShapeDesc& shape = desc->shapes[i];
        if (shape.object >= 0 || !shape_meshes[i]) {
            continue;
        }
        Node node;
        node.name = fmt::format("shape {:03}", i);
        node.parent = root_id;
        node.local_transform = shape.ctm;
        const NodeID node_id = add_node(node);
        add_mesh_instance(*shape_meshes[i], node_id);
        expand_by(shape_aabbs[i], shape.ctm);
    }

    // 2. object instances: parent carries the instance CTM, children the shape transform
    // relative to the CTM at ObjectBegin.
    for (const pbrt::InstanceDesc& instance : desc->instances) {
        const pbrt::ObjectDesc& object = desc->objects[instance.object];
        Node parent;
        parent.name = object.name;
        parent.parent = root_id;
        parent.local_transform = instance.ctm;
        const NodeID parent_id = add_node(parent);

        const float4x4 inv_begin = inverse(object.begin_ctm);
        for (const int32_t shape_index : object.shape_indices) {
            if (!shape_meshes[shape_index]) {
                continue;
            }
            Node child;
            child.parent = parent_id;
            child.local_transform = mul(inv_begin, desc->shapes[shape_index].ctm);
            const NodeID child_id = add_node(child);
            add_mesh_instance(*shape_meshes[shape_index], child_id);
            expand_by(shape_aabbs[shape_index], mul(instance.ctm, child.local_transform));
        }
    }
}

// ---------------------------------------------------------------------------
// Environment and camera
// ---------------------------------------------------------------------------

void PBRTScene::load_env(const CommandBufferHandle& cmd) {
    if (desc->infinite_lights.empty()) {
        return;
    }
    if (desc->infinite_lights.size() > 1) {
        SPDLOG_WARN("PBRTScene: multiple infinite lights, using the first");
    }
    const pbrt::InfiniteLightDesc& light = desc->infinite_lights[0];

    TextureHandle texture;
    if (!light.filename.empty()) {
        const std::filesystem::path path = base_dir / light.filename;
        if (path.extension() == ".exr") {
            warn_once("exr", "EXR images are unsupported, skipping " + light.filename);
            return;
        }
        try {
            ImageInfo info;
            const BlobHandle blob = image_load_f32(path, info, 4);
            if (info.width != info.height) {
                SPDLOG_WARN("PBRTScene: infinite light image is not square: {}", path.string());
            }
            texture = get_allocator()->create_texture_from_rgba32f(
                cmd, blob->get_data<float>(), static_cast<uint32_t>(info.width),
                static_cast<uint32_t>(info.height), vk::SamplerAddressMode::eClampToEdge,
                vk::Filter::eLinear, vk::Filter::eLinear, "pbrt_env");
        } catch (const std::exception& e) {
            SPDLOG_WARN("PBRTScene: failed to load env map '{}': {}", path.string(), e.what());
            return;
        }
    } else {
        const float3 radiance = light.radiance.value_or(float3(1));
        const float rgba[4] = {radiance.x, radiance.y, radiance.z, 1.f};
        texture = get_allocator()->create_texture_from_rgba32f(
            cmd, rgba, 1, 1, vk::SamplerAddressMode::eClampToEdge, vk::Filter::eLinear,
            vk::Filter::eLinear, "pbrt_env_const");
    }
    cmd->barrier(texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));

    const auto env = std::make_shared<EqualAreaOctEnvMap>(texture);
    env->set_intensity(light.scale);
    // pbrt evaluates the image at inverse(light CTM) * world direction.
    env->set_base_transform(float3x3(inverse(light.ctm)));
    set_env(env);
}

void PBRTScene::load_camera() {
    if (desc->camera) {
        const float4x4 camera_to_world = inverse(desc->camera->world_to_camera);
        const float3 eye(camera_to_world[0][3], camera_to_world[1][3], camera_to_world[2][3]);
        // pbrt camera space looks down +z.
        const float3 forward =
            normalize(float3(camera_to_world[0][2], camera_to_world[1][2], camera_to_world[2][2]));
        const float3 up =
            normalize(float3(camera_to_world[0][1], camera_to_world[1][1], camera_to_world[2][1]));

        const float aspect =
            static_cast<float>(desc->film_width) / static_cast<float>(desc->film_height);
        // fov applies to the shorter image axis.
        const float fov = radians(desc->camera->fov);
        const float fov_y = desc->film_width >= desc->film_height
                                ? fov
                                : 2.f * std::atan(std::tan(0.5f * fov) / aspect);

        const auto camera =
            std::make_shared<Camera>(eye, eye + forward, up, fov_y, aspect, 0.01f, 1000.f);
        camera->set_resolution(vk::Extent3D{static_cast<uint32_t>(desc->film_width),
                                            static_cast<uint32_t>(desc->film_height), 1});
        add_camera(camera);
    }

    if (get_cameras().empty()) {
        SPDLOG_INFO("PBRTScene: no camera in file, adding default camera");
        add_camera(std::make_shared<Camera>(float3(3, 3, 3), float3(0, 0, 0), get_up(),
                                            radians(60.f), 1920.f / 1080.f, 0.01f, 1000.f));
        AABB& aabb = get_aabb();
        if (aabb.is_valid()) {
            get_active_camera()->look_at_bounding_box(aabb);
        }
    }

    for (const CameraHandle& cam : get_cameras()) {
        cam->set_jitter_sequence(Camera::JitterSequence::R2);
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

void PBRTScene::release_textures(const CommandBufferHandle& cmd) {
    // Defer texture destruction to pool reset so any in-flight frame keeps its bindings valid.
    const auto release = [&](const TextureID id) {
        if (id != TextureID(-1)) {
            cmd->keep_until_pool_reset(get_texture_manager()->get_texture(id));
            get_texture_manager()->remove_texture(id);
        }
    };
    for (const auto& [key, slot] : texture_slots) {
        release(slot.id_srgb);
        release(slot.id_linear);
    }
    // Baked textures (checkerboard) are referenced only by the resolve cache.
    for (const auto& [key, resolved] : resolved_textures) {
        if (resolved.texture == TextureID(-1)) {
            continue;
        }
        bool owned_by_slot = false;
        for (const auto& [file, slot] : texture_slots) {
            if (slot.id_srgb == resolved.texture || slot.id_linear == resolved.texture) {
                owned_by_slot = true;
                break;
            }
        }
        if (!owned_by_slot) {
            release(resolved.texture);
        }
    }
    texture_slots.clear();
    resolved_textures.clear();
}

void PBRTScene::load(const CommandBufferHandle& cmd, const std::filesystem::path& path) {
    release_textures(cmd);
    material_cache.clear();
    shape_meshes.clear();
    shape_aabbs.clear();
    warned.clear();

    clear_geometry();
    get_material_system()->clear();
    desc.reset();

    SPDLOG_INFO("PBRTScene: loading {}", path.string());
    std::unique_ptr<pbrt::PBRTSceneDesc> parsed;
    try {
        parsed = pbrt::parse_pbrt_file(path);
    } catch (const std::exception& e) {
        // Leave desc null so is_ready() reports false and update() bails out cleanly.
        SPDLOG_ERROR("PBRTScene: failed to load '{}': {}", path.string(), e.what());
        return;
    }
    desc = std::move(parsed);
    base_dir = desc->base_dir;

    build_scene(cmd);

    load_env(cmd);

    load_camera();

    SPDLOG_INFO("PBRTScene: loaded '{}' shapes: {}, materials: {}, instances: {}, textures: {}",
                path.filename().string(), desc->shapes.size(), desc->materials.size(),
                desc->instances.size(), desc->textures.size());
}

} // namespace merian
