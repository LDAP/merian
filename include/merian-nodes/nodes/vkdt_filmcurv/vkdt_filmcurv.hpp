#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian {

struct VKDTFilmcurvePushConstant {
    float brightness{1.};
    float contrast{1.};
    float bias{0.};
    int32_t colourmode{1};
};

class VKDTFilmcurv : public TypedPCAbstractCompute<VKDTFilmcurvePushConstant> {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

  public:
    VKDTFilmcurv();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    const VKDTFilmcurvePushConstant& get_typed_push_constant(GraphRun& run,
                                                             const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    VulkanEntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    std::optional<vk::Format> output_format = std::nullopt;

    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read();
    vk::Extent3D extent;

    VulkanEntryPointHandle shader;
    SpecializationInfoHandle spec_info;

    VKDTFilmcurvePushConstant pc;
};

} // namespace merian
