#pragma once

#include "merian-shaders/scene/scene.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

#include <filesystem>
#include <memory>

namespace tinygltf {
class Model;
class Node;
} // namespace tinygltf

namespace merian {

class GLTFScene : public Scene {
  public:
    GLTFScene(const ShaderCompileContextHandle& compile_context,
              const ContextHandle& context,
              const ResourceAllocatorHandle& allocator,
              const ShaderObjectAllocatorHandle& obj_allocator,
              const MaterialSystemHandle& material_system);

    ~GLTFScene() override;

    // Load a glTF file (.gltf or .glb)
    void load(const CommandBufferHandle& cmd, const std::filesystem::path& path);

    float3 get_up() override {
        return float3(0, 1, 0);
    }

    // If true, generate mipmaps for baseColor and emissive textures even when the glTF sampler
    // does not request a mipmap minFilter. Has no effect on normal/MR/occlusion textures.
    void set_force_mipmaps_color(bool enable) {
        force_mipmaps_color = enable;
    }

  private:
    void load_materials(const CommandBufferHandle& cmd);

    void load_meshes();

    void load_node(int gltf_node_index, NodeID parent_id);

    void load_cameras();

    void compute_aabb();

    // Returns the GPU TextureID for the given glTF texture index, sampling in the requested
    // color space. Lazily uploads on first request. If the same image is needed in both color
    // spaces (rare — same texture used as both color and data), the data is uploaded twice
    // rather than aliasing the image with VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, which would
    // hurt sampling performance for every texture.
    TextureID
    get_or_load_texture(const CommandBufferHandle& cmd, int gltf_tex_idx, bool linear);

    // Owned so GLTFMesh can reference buffer data directly.
    std::unique_ptr<tinygltf::Model> model;

    // glTF material index -> MaterialID
    std::vector<MaterialID> material_map;

    // glTF node index -> NodeID
    std::vector<NodeID> node_map;

    // glTF mesh index -> MeshIDs (one for each gltf primitive)
    std::vector<std::vector<MeshID>> mesh_map;

    // One sampler per glTF sampler index.
    std::vector<SamplerHandle> gltf_samplers;
    // Used when a glTF texture has no sampler set (tex.sampler == -1).
    SamplerHandle default_gltf_sampler;
    // Per glTF sampler index: did the sampler request mipmaps via *_MIPMAP_* minFilter?
    std::vector<bool> gltf_sampler_wants_mipmaps;

    struct GltfTextureSlot {
        TextureID id_srgb = TextureID(-1);
        TextureID id_linear = TextureID(-1);
    };
    // One entry per glTF texture index; populated lazily by get_or_load_texture.
    std::vector<GltfTextureSlot> texture_slots;

    MaterialModelID diffuse_type_id;

    bool force_mipmaps_color = true;
};

using GLTFSceneHandle = std::shared_ptr<GLTFScene>;

} // namespace merian
