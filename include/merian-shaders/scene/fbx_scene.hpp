#pragma once

#include "merian-shaders/scene/scene.hpp"

#include <filesystem>

struct ufbx_scene;
struct ufbx_node;
struct ufbx_mesh;
struct ufbx_texture;

namespace merian {

class FBXScene : public Scene {
  public:
    FBXScene(const ShaderCompileContextHandle& compile_context,
             const ContextHandle& context,
             const ResourceAllocatorHandle& allocator,
             const MaterialSystemHandle& material_system);

    ~FBXScene() override;

    // Load an FBX file (.fbx)
    void load(const CommandBufferHandle& cmd, const std::filesystem::path& path);

    float3 get_up() override {
        return float3(0, 1, 0);
    }

    bool is_ready() const override {
        return scene != nullptr;
    }

  private:
    void load_materials(const CommandBufferHandle& cmd);

    void load_meshes();

    void load_node(const ufbx_node* node, NodeID parent_id);

    void load_cameras();

    void compute_aabb();

    void free_scene();

    TextureID
    get_or_load_texture(const CommandBufferHandle& cmd, const ufbx_texture* tex, bool linear);

    // Owned so geometry/material loading can reference ufbx data directly.
    ufbx_scene* scene = nullptr;

    // Directory of the loaded file, for resolving relative texture paths.
    std::filesystem::path base_dir;

    // ufbx material typed_id -> MaterialID
    std::vector<MaterialID> material_map;
    // ufbx material typed_id -> whether it is fully opaque (no base-color alpha cutout).
    std::vector<uint8_t> material_opaque;
    // Used for mesh parts without an assigned material.
    MaterialID default_material_id{};

    // ufbx node typed_id -> NodeID
    std::vector<NodeID> node_map;

    // ufbx mesh typed_id -> MeshIDs (one per material part)
    std::vector<std::vector<MeshID>> mesh_map;

    struct TextureSlot {
        TextureID id_srgb = TextureID(-1);
        TextureID id_linear = TextureID(-1);
        bool has_alpha = false;
    };
    // One entry per ufbx texture typed_id; populated lazily by get_or_load_texture.
    std::vector<TextureSlot> texture_slots;

    MaterialModelID pbrt_type_id;
};

using FBXSceneHandle = std::shared_ptr<FBXScene>;

} // namespace merian
