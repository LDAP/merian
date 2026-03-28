#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/shader/shader_object.hpp"
#include "merian/shader/shader_object_allocator.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/slang_program.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

namespace merian {

/**
 * @brief Test node for the ShaderObject/ShaderCursor API.
 *
 * A simple compute node that takes a sampled image input and writes a packed GBuffer output.
 * This serves as a placeholder/test for the Slang parameter binding system.
 */
class GBufferRTNode : public Node {

  public:
    GBufferRTNode();

    ~GBufferRTNode() override = default;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

  private:
    ContextHandle context;
    ResourceAllocatorHandle resource_allocator;

    // Connectors
    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read();
    ManagedVkImageOutHandle con_out;

    // Slang program + pipeline
    ShaderCompileContextHandle compile_context;
    SlangProgramHandle program;
    SlangProgramEntryPointHandle entry_point;
    PipelineHandle pipeline;

    // ShaderObject for parameter binding
    ShaderObjectHandle params;
    std::shared_ptr<FrameCachingShaderObjectAllocator> obj_allocator;

    // Test sub-objects created explicitly via write(ShaderObjectHandle)
    ShaderObjectHandle manual_cb_obj;
    ShaderObjectHandle manual_pb_obj;

    // Reassignment test: replaced after N frames
    ShaderObjectHandle replace_cb_obj;
    uint32_t frame_count = 0;
};

} // namespace merian
