#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"

#include "merian/vk/pipeline/specialization_info.hpp"

#include <optional>

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

    NodeStatusFlags on_connected(const NodeConnectedInfo& info) override;

    void process(GraphRun& run, const NodeIO& io) override;

  private:
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;

    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    ManagedVkBufferOutHandle con_mean;

    PushConstant pc;

    Versioned<SpecializationInfo> image_to_buffer_spec;
    Versioned<SpecializationInfo> reduce_buffer_spec;

    std::optional<ComputeKernel> image_to_buffer_kernel;
    std::optional<ComputeKernel> reduce_buffer_kernel;
};

} // namespace merian
