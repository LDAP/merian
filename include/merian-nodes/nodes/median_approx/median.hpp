#pragma once

#include "merian-nodes/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/shader/entry_point.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

namespace merian {

class MedianApproxNode : public Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        float min = 0;
        float max = 1000;
    };

  public:
    MedianApproxNode();

    virtual ~MedianApproxNode();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    ContextHandle context;
    int component;

    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read();
    ManagedVkBufferOutHandle con_median;
    ManagedVkBufferOutHandle con_histogram;

    PushConstant pc;

    EntryPointHandle histogram;
    EntryPointHandle reduce;

    PipelineLayoutHandle pipe_layout;

    PipelineHandle pipe_histogram;
    PipelineHandle pipe_reduce;
};

} // namespace merian
