#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"

#include "merian/vk/pipeline/specialization_info.hpp"

#include <optional>

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

    NodeStatusFlags on_connected(const NodeConnectedInfo& info) override;

    void process(GraphRun& run, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    void make_spec_info();

    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;
    int component = 0;

    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    ManagedVkBufferOutHandle con_median;
    ManagedVkBufferOutHandle con_histogram;

    PushConstant pc;

    Versioned<SpecializationInfo> spec_info;

    std::optional<ComputeKernel> histogram_kernel;
    std::optional<ComputeKernel> reduce_kernel;
};

} // namespace merian
