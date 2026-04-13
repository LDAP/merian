#pragma once

#include "merian-shaders/scene/scene.hpp"

#include <filesystem>

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

    // Load a glTF file (.gltf or .glb)
    void load(const CommandBufferHandle& cmd, const std::filesystem::path& path);

  private:
    void load_materials(const CommandBufferHandle& cmd,
                        const tinygltf::Model& model,
                        const std::filesystem::path& base_dir);
    void load_node(const tinygltf::Model& model, int gltf_node_index, NodeID parent_id);

    // Map from glTF material index to MaterialID
    std::vector<MaterialID> material_map;
    // Map from glTF node index to merian NodeID
    std::vector<NodeID> node_map;

    MaterialModelID diffuse_type_id;
};

using GLTFSceneHandle = std::shared_ptr<GLTFScene>;

} // namespace merian
