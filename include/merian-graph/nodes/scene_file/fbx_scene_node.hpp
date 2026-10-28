#pragma once

#include "merian-graph/nodes/scene_file/scene_file_node.hpp"

namespace merian {

class FBXSceneNode : public SceneFileNode {
  public:
    FBXSceneNode();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    NodeStatusFlags properties(Properties& config) override;

  protected:
    SceneHandle create_scene(const ShaderCompileContextHandle& compile_context,
                             const ContextHandle& context,
                             const ResourceAllocatorHandle& allocator,
                             const MaterialSystemHandle& material_system) override;

    void load_file(Scene& scene,
                   const CommandBufferHandle& cmd,
                   const std::filesystem::path& path) override;

  private:
    bool orm_specular_textures = true;
};

} // namespace merian
