#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/nodes/compute_node/compute_node.hpp"

namespace merian {

class TAA : public AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        // higher value means more temporal reuse
        float temporal_alpha;
        int clamp_method;
        VkBool32 enable_mv;
    };

  public:
    TAA();

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorDescriptor> describe_inputs() override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    const void* get_push_constant([[maybe_unused]] GraphRun& run,
                                  [[maybe_unused]] const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count([[maybe_unused]] const NodeIO& io) const noexcept override;

    SlangCompositionHandle create_composition() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    const bool inverse_motion = false;

    VkSampledImageInHandle con_src = VkSampledImageIn::create();
    VkSampledImageInHandle con_mv = VkSampledImageIn::create();

    PushConstant pc;
    uint32_t width{};
    uint32_t height{};
};

} // namespace merian
