#pragma once

#include "merian-nodes/connectors/managed_vk_buffer_out.hpp"
#include "merian-nodes/connectors/vk_texture_in.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian_nodes {

class MeanToBuffer : public Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;
    static constexpr uint32_t workgroup_size = local_size_x * local_size_y;

    struct PushConstant {
        uint32_t divisor;

        int size;
        int offset;
        int count;
    };

  public:
    MeanToBuffer(const ContextHandle& context);

    ~MeanToBuffer();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

  private:
    const ContextHandle context;

    VkTextureInHandle con_src = VkTextureIn::compute_read("src");
    ManagedVkBufferOutHandle con_mean;

    PushConstant pc;

    ShaderModuleHandle image_to_buffer_shader;
    ShaderModuleHandle reduce_buffer_shader;

    PipelineHandle image_to_buffer;
    PipelineHandle reduce_buffer;
};

} // namespace merian_nodes
