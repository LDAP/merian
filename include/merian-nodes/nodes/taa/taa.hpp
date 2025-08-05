#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian_nodes {

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
    TAA(const ContextHandle& context);

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    SpecializationInfoHandle get_specialization_info(const NodeIO& io) noexcept override;

    const void* get_push_constant([[maybe_unused]] GraphRun& run,
                                  [[maybe_unused]] const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count([[maybe_unused]] const NodeIO& io) const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    const bool inverse_motion = false;
    ShaderModuleHandle shader;
    SpecializationInfoHandle spec_info;

    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read("src");
    VkSampledImageInHandle con_mv = VkSampledImageIn::compute_read("mv", 0, true);

    PushConstant pc;
    uint32_t width{};
    uint32_t height{};
};

} // namespace merian_nodes
