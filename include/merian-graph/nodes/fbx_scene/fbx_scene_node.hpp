#pragma once

#include "merian-graph/connectors/ptr_out.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-shaders/scene/scene.hpp"

#ifdef MERIAN_UFBX_ENABLED
#include "merian-shaders/scene/fbx_scene.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include <filesystem>
#endif

namespace merian {

class FBXSceneNode : public Node {

  public:
    FBXSceneNode();

    ~FBXSceneNode() override = default;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
#ifdef MERIAN_UFBX_ENABLED
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;
    TextureManagerHandle texture_manager;
    MaterialSystemHandle material_system;

    FBXSceneHandle scene;
    std::filesystem::path file_path;
    bool needs_load = false;

#endif

    PtrOutHandle<Scene> con_scene = PtrOut<Scene>::create(true);
};

} // namespace merian
