#pragma once

#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/ptr_out.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-scene/scene.hpp"
#include "merian/utils/camera/camera_controller.hpp"
#include "merian/utils/input_controller.hpp"

#ifdef MERIAN_TINYGLTF_ENABLED
#include "merian-scene/gltf_scene.hpp"
#include "merian/shader/shader_compile_context.hpp"
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

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
#ifdef MERIAN_TINYGLTF_ENABLED
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;
    TextureManagerHandle texture_manager;
    MaterialSystemHandle material_system;

    GLTFSceneHandle scene;
    std::filesystem::path file_path;

#endif

    PtrOutHandle<Scene> con_scene = PtrOut<Scene>::create(true);

    PtrInHandle<InputController> con_controller = PtrIn<InputController>::create(0, true);
    std::shared_ptr<CameraController> cam_controller = std::make_shared<CameraController>();
    std::weak_ptr<InputController> registered_controller;
};

} // namespace merian
