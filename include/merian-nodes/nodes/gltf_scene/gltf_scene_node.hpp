#pragma once

#include "merian-nodes/connectors/ptr_out.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-shaders/scene/scene.hpp"

#ifdef MERIAN_TINYGLTF_ENABLED
#include "merian-shaders/scene/gltf_scene.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include <filesystem>
#endif

namespace merian {

class GLTFSceneNode : public Node {

  public:
    GLTFSceneNode();

    ~GLTFSceneNode() override = default;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
#ifdef MERIAN_TINYGLTF_ENABLED
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;
    std::shared_ptr<FrameCachingShaderObjectAllocator> obj_allocator;
    TextureManagerHandle texture_manager;
    MaterialSystemHandle material_system;

    GLTFSceneHandle scene;
    std::filesystem::path file_path;
    bool needs_load = false;

#endif

    PtrOutHandle<Scene> con_scene = PtrOut<Scene>::create(true);
};

} // namespace merian
