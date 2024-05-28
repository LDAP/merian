#pragma once

#include "merian-nodes/connectors/vk_buffer_out.hpp"
#include "merian-nodes/connectors/vk_image_in.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian_nodes {

class MedianApproxNode : public Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        float min = 0;
        float max = 1000;
    };

  public:
    MedianApproxNode(const SharedContext context, const int component = 0);

    virtual ~MedianApproxNode();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    NodeStatusFlags on_connected(const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

    NodeStatusFlags configuration(Configuration& config) override;

  private:
    const SharedContext context;
    const int component;

    VkImageInHandle con_src = VkImageIn::compute_read("src");
    VkBufferOutHandle con_median;
    VkBufferOutHandle con_histogram;

    PushConstant pc;

    ShaderModuleHandle histogram;
    ShaderModuleHandle reduce;

    PipelineHandle pipe_histogram;
    PipelineHandle pipe_reduce;
};

} // namespace merian_nodes
