#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"

#include "merian-graph/graph/node.hpp"
#include "merian-graph/nodes/compute_node/compute_kernel.hpp"

#include "merian/vk/pipeline/specialization_info.hpp"

#include <optional>

namespace merian {

class Bloom : public Node {
  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        float threshold = 10.0;
        float strength = 0.001;
    };

  public:
    Bloom();

    virtual ~Bloom();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected(const NodeConnectedInfo& info) override;

    void process(GraphRun& run, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    ContextHandle context;
    ResourceAllocatorHandle allocator;
    ShaderCompileContextHandle compile_context;

    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    ManagedVkImageOutHandle con_out;
    ManagedVkImageOutHandle con_interm;

    PushConstant pc;
    Versioned<SpecializationInfo> spec_info;

    std::optional<ComputeKernel> separate_kernel;
    std::optional<ComputeKernel> composite_kernel;

    int32_t mode = 0;
};

} // namespace merian
