#pragma once

#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/connectors/ptr_out.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-shaders/gbuffer.hpp"
#include "merian-shaders/scene/scene.hpp"

#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

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
    PtrOutHandle<GBuffer> con_gbuffer;

    // Image connectors (managed by graph, also exposed individually)
    ManagedVkImageOutHandle con_denoiser;
    ManagedVkImageOutHandle con_hit_info;
    ManagedVkImageOutHandle con_mv;
    ManagedVkImageOutHandle con_albedo;

    // Resolution
    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};

    // Slang program + pipeline
    SlangProgramHandle program;
    SlangProgramEntryPointHandle entry_point;
    PipelineHandle pipeline;

    // ShaderObject for GBuffer parameter
    GBufferHandle gbuffer_obj;
    std::shared_ptr<FrameCachingShaderObjectAllocator> obj_allocator;
};

} // namespace merian
