#pragma once

#include "merian-graph/nodes/scene_file/scene_file_node.hpp"

namespace merian {

class PBRTSceneNode : public SceneFileNode {
  public:
    PBRTSceneNode();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

  protected:
    SceneHandle create_scene(const ShaderCompileContextHandle& compile_context,
                             const ContextHandle& context,
                             const ResourceAllocatorHandle& allocator,
                             const MaterialSystemHandle& material_system) override;

    void load_file(Scene& scene,
                   const CommandBufferHandle& cmd,
                   const std::filesystem::path& path) override;
};

} // namespace merian
