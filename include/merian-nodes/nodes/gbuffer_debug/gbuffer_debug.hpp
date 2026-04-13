#pragma once

#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-nodes/nodes/gbuffer_rt/gbuffer_resource.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

namespace merian {

class GBufferDebugNode : public Node {

  public:
    enum FieldSelect : int {
        FIELD_NORMAL = 0,
        FIELD_LINEAR_Z = 1,
        FIELD_GRAD_Z = 2,
        FIELD_DELTA_Z = 3,
        FIELD_MOTION_VECTORS = 4,
        FIELD_INSTANCE_ID = 5,
        FIELD_PRIMITIVE_ID = 6,
        FIELD_BARYCENTRICS = 7,
    };

    GBufferDebugNode();

    ~GBufferDebugNode() override = default;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& io_layout) override;

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
    PtrInHandle<GBufferResource> con_gbuffer = PtrIn<GBufferResource>::create();
    ManagedVkImageOutHandle con_output;

    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};
    FieldSelect selected_field = FIELD_NORMAL;

    // Slang program + pipeline
    SlangProgramHandle program;
    SlangProgramEntryPointHandle entry_point;
    PipelineHandle pipeline;

    ShaderObjectHandle debug_gbuffer_obj;
    std::shared_ptr<FrameCachingShaderObjectAllocator> obj_allocator;
};

} // namespace merian
