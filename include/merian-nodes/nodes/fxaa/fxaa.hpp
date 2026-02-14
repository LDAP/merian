#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian {

class FXAA : public AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        int32_t enable = 1;
        float fxaaQualitySubpix = 0.5;
        float fxaaQualityEdgeThreshold = 0.166;
        float fxaaQualityEdgeThresholdMin = 0.0833;
    };

  public:
    FXAA();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    const void* get_push_constant(GraphRun& run, const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    VulkanEntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    vk::Extent3D extent;
    PushConstant pc;
    SpecializationInfoHandle spec_info;
    VulkanEntryPointHandle shader;

    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read();
};

} // namespace merian
