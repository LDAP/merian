#pragma once

#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/ptr_out.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-shaders/scene/scene.hpp"
#include "merian/utils/camera/camera_controller.hpp"
#include "merian/utils/input_controller.hpp"

#include <filesystem>

namespace merian {

class SceneFileNode : public Node {
  public:
    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    [[nodiscard]] NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                               const NodeIO& io,
                                               const NodeConnectionInfo& info,
                                               Submission& submission) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

    // Null until the node is initialized.
    const SceneHandle& get_scene() const;

  protected:
    explicit SceneFileNode(std::string file_description);

    virtual SceneHandle create_scene(const ShaderCompileContextHandle& compile_context,
                                     const ContextHandle& context,
                                     const ResourceAllocatorHandle& allocator,
                                     const MaterialSystemHandle& material_system) = 0;

    // scene is the one create_scene returned.
    virtual void
    load_file(Scene& scene, const CommandBufferHandle& cmd, const std::filesystem::path& path) = 0;

    void reload();

  private:
    const std::string file_description;

    ContextHandle context;
    SceneHandle scene;
    std::filesystem::path file_path;

    PtrOutHandle<Scene> con_scene = PtrOut<Scene>::create();

    PtrInHandle<InputController> con_controller = PtrIn<InputController>::create();
    std::shared_ptr<CameraController> cam_controller = std::make_shared<CameraController>();
    std::weak_ptr<InputController> registered_controller;
};

} // namespace merian
