#pragma once

#include "merian-nodes/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/shader/entry_point.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

namespace merian {

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
    MeanToBuffer();

    ~MeanToBuffer();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

  private:
    ContextHandle context;

    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read();
    ManagedVkBufferOutHandle con_mean;

    PushConstant pc;

    EntryPointHandle image_to_buffer_shader;
    EntryPointHandle reduce_buffer_shader;

    PipelineHandle image_to_buffer;
    PipelineHandle reduce_buffer;
};

} // namespace merian
