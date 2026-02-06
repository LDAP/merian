#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

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
    };

  public:
    Tonemap();

    ~Tonemap();

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    const void* get_push_constant(GraphRun& run, const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    VulkanEntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    void make_spec_info();

    std::optional<vk::Format> output_format = std::nullopt;

    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read("src");

    vk::Extent3D extent;
    PushConstant pc;
    VulkanEntryPointHandle shader;
    SpecializationInfoHandle spec_info;

    int32_t tonemap = 0;
    int32_t alpha_mode = 0;
    int32_t clamp_output = 1;
};

} // namespace merian
