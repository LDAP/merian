#pragma once

#include "merian-shaders/scene/scene.hpp"

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

  private:
    void load_materials(const CommandBufferHandle& cmd);

    void load_meshes();

    void load_node(int gltf_node_index, NodeID parent_id);

    void load_cameras();

    void compute_aabb();

    // Owned so GLTFMesh can reference buffer data directly.
    std::unique_ptr<tinygltf::Model> model;

    // glTF material index -> MaterialID
    std::vector<MaterialID> material_map;

    // glTF node index -> NodeID
    std::vector<NodeID> node_map;

    // glTF mesh index -> MeshIDs (one for each gltf primitive)
    std::vector<std::vector<MeshID>> mesh_map;

    MaterialModelID diffuse_type_id;
};

using GLTFSceneHandle = std::shared_ptr<GLTFScene>;

} // namespace merian
