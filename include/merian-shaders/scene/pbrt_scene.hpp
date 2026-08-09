#pragma once

#include "merian-shaders/scene/scene.hpp"

#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace merian {

namespace pbrt {
struct PBRTSceneDesc;
struct ShapeDesc;
class ParamDict;
} // namespace pbrt

class PBRTScene : public Scene {
  public:
    PBRTScene(const ShaderCompileContextHandle& compile_context,
              const ContextHandle& context,
              const ResourceAllocatorHandle& allocator,
              const MaterialSystemHandle& material_system);

    ~PBRTScene() override;

    // Load a pbrt-v4 scene file (.pbrt, optionally gzip-compressed).
    void load(const CommandBufferHandle& cmd, const std::filesystem::path& path);

    float3 get_up() override {
        return float3(0, 1, 0);
    }

    bool is_ready() const override {
        return desc != nullptr;
    }

  private:
    // pbrt texture references collapse to a constant factor and at most one sampled texture.
    struct Resolved {
        float3 factor{1, 1, 1};
        TextureID texture{TextureID(-1)};
        bool has_alpha = false;
    };

    struct TextureSlot {
        TextureID id_srgb = TextureID(-1);
        TextureID id_linear = TextureID(-1);
        bool has_alpha = false;
    };

    struct MaterialBuild; // OpenPBRMaterial + derived mesh flags, defined in the .cpp

    void release_textures(const CommandBufferHandle& cmd);
    void build_scene(const CommandBufferHandle& cmd);
    void load_env(const CommandBufferHandle& cmd);
    void load_camera();

    std::optional<MeshID> build_shape_mesh(const CommandBufferHandle& cmd, size_t shape_index);

    MaterialID material_for_shape(const CommandBufferHandle& cmd,
                                  const pbrt::ShapeDesc& shape,
                                  MeshFlags& out_flags);
    MaterialBuild
    convert_material(const CommandBufferHandle& cmd, int32_t material_index, int depth);
    void apply_roughness(const CommandBufferHandle& cmd,
                         const pbrt::ParamDict& params,
                         const std::string& prefix,
                         float& out_roughness,
                         TextureID& out_texture);

    Resolved resolve_color_param(const CommandBufferHandle& cmd,
                                 const pbrt::ParamDict& params,
                                 const char* name,
                                 const float3& fallback,
                                 bool srgb);
    Resolved resolve_texture_ref(const CommandBufferHandle& cmd,
                                 const std::string& name,
                                 bool srgb,
                                 int depth);
    Resolved resolve_texture_input(const CommandBufferHandle& cmd,
                                   const pbrt::ParamDict& params,
                                   const char* name,
                                   const float3& fallback,
                                   bool srgb,
                                   int depth);
    TextureID load_image_texture(const CommandBufferHandle& cmd,
                                 const std::string& filename,
                                 bool srgb,
                                 bool* out_has_alpha);

    void warn_once(const std::string& key, const std::string& message);

    std::unique_ptr<pbrt::PBRTSceneDesc> desc;
    std::filesystem::path base_dir;

    // image file path -> uploaded textures (per color space)
    std::unordered_map<std::string, TextureSlot> texture_slots;
    // named pbrt texture (+ color space) -> resolved factor/texture
    std::unordered_map<std::string, Resolved> resolved_textures;
    // emission is baked into the material, so emissive uses of a shared named material copy
    struct CachedMaterial {
        MaterialID id;
        MeshFlags flags;
    };
    std::map<std::tuple<int32_t, float, float, float, bool, int32_t>, CachedMaterial>
        material_cache;

    std::vector<std::optional<MeshID>> shape_meshes;
    std::vector<AABB> shape_aabbs;

    std::set<std::string> warned;
};

using PBRTSceneHandle = std::shared_ptr<PBRTScene>;

} // namespace merian
