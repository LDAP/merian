#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"

#include "merian-nodes/graph/node.hpp"

#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/shader/entry_point.hpp"

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
    Bloom(const ContextHandle& context);

    virtual ~Bloom();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    const ContextHandle context;

    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read("src");
    ManagedVkImageOutHandle con_out;
    ManagedVkImageOutHandle con_interm;

    PushConstant pc;

    EntryPointHandle separate_module;
    EntryPointHandle composite_module;

    PipelineHandle separate;
    PipelineHandle composite;

    int32_t mode = 0;
};

} // namespace merian_nodes
