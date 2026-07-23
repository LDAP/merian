#pragma once

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/connectors/ptr_in.hpp"
#include "merian-graph/connectors/shader_object_out.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/objects/gbuffer_object.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_composition.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing.hpp"
#include "merian/vk/raytrace/shader_binding_table.hpp"

#include <array>

namespace merian {

class GBufferRTNode : public Node {

  public:
    GBufferRTNode();

    ~GBufferRTNode() override = default;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeConnectedInfo& info) override;

    void process(GraphRun& run, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    void update_gbuffer_constants();

    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    // Connectors
    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    ShaderObjectOutHandle<GBufferObject> con_gbuffer;
    ManagedVkImageOutHandle con_emission;

    // Resolution
    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};

    std::array<bool, 8> mask_enabled{true, true, true, true, true, true, true, true};

    bool emission_connected = true;

    // Slang program + pipeline; rebuilt when the scene composition changes.
    SlangCompositionHandle composition;
    Versioned<SlangProgram> program;
    Versioned<SlangProgramEntryPoint> entry_point;
    Versioned<RayTracingPipeline> pipeline;
    Versioned<ShaderBindingTable> sbt;
    Versioned<ShaderObject> globals_obj;

    std::shared_ptr<FrameCachingShaderObjectAllocator> obj_allocator;
};

} // namespace merian
