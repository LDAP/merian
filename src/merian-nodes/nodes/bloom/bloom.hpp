#pragma once

#include "merian-nodes/connectors/vk_image_in.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian_nodes {

class Bloom : public Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        float threshold = 10.0;
        float strength = 0.001;
    };

  public:
    Bloom(const SharedContext context);

    virtual ~Bloom();

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

    VkImageInHandle con_src = VkImageIn::compute_read("src");
    VkImageOutHandle con_out;
    VkImageOutHandle con_interm;

    PushConstant pc;

    ShaderModuleHandle separate_module;
    ShaderModuleHandle composite_module;

    PipelineHandle separate;
    PipelineHandle composite;

    int32_t mode = 0;
};

} // namespace merian_nodes
