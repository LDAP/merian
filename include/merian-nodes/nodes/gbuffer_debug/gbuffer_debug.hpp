#pragma once

#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-shaders/gbuffer.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

namespace merian {

class GBufferDebugNode : public Node {

  public:
    GBufferDebugNode();

    ~GBufferDebugNode() override = default;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;
    ShaderCompileContextHandle compile_context;

    // Connectors
    PtrInHandle<Scene> con_scene = PtrIn<Scene>::create();
    PtrInHandle<GBuffer> con_gbuffer = PtrIn<GBuffer>::create();
    ManagedVkImageOutHandle con_output;

    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};
    int32_t selected_field = 0;

    // Slang program + pipeline
    SlangProgramHandle program;
    SlangProgramEntryPointHandle entry_point;
    PipelineHandle pipeline;

    ShaderObjectHandle params;
    std::shared_ptr<FrameCachingShaderObjectAllocator> obj_allocator;
};

} // namespace merian
