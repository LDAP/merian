#pragma once

#include "merian-nodes/connectors/ptr_out.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-shaders/scene/gltf_scene.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object_allocator.hpp"

#include <filesystem>

namespace merian {

class GLTFSceneNode : public Node {

  public:
    GLTFSceneNode();

    ~GLTFSceneNode() override = default;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;
    ShaderObjectAllocatorHandle obj_allocator;
    TextureManagerHandle texture_manager;
    MaterialSystemHandle material_system;

    GLTFSceneHandle scene;
    std::filesystem::path file_path;
    bool needs_load = false;
    uint32_t frame = 0;

    PtrOutHandle<Scene> con_scene = PtrOut<Scene>::create(true);
};

} // namespace merian
