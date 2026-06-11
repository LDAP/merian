#pragma once

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/nodes/compute_node/compute_node.hpp"

namespace merian {

class Tonemap : public AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        float param1 = 1.0;
        float param2 = 1.0;
        float param3 = 1.0;
        float param4 = 1.0;
        float param5 = 1.0;

        float perceptual_exponent = 2.2;

        // AgX ASC CDL look (defaults to the "Default" preset)
        float agx_slope_r = 1.0;
        float agx_slope_g = 1.0;
        float agx_slope_b = 1.0;
        float agx_offset = 0.0;
        float agx_power = 1.0;
        float agx_sat = 1.0;
    };

  public:
    Tonemap();

    ~Tonemap();

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
    void make_spec_info();

    void apply_agx_look();

    std::optional<vk::Format> output_format = std::nullopt;

    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read();

    vk::Extent3D extent;
    PushConstant pc;
    VulkanEntryPointHandle shader;
    SpecializationInfoHandle spec_info;

    int32_t tonemap = 0;
    int32_t alpha_mode = 0;
    int32_t clamp_output = 1;
    int32_t agx_look = 0;
};

} // namespace merian
