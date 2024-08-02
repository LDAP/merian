#pragma once

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian_nodes {

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
    FXAA(const ContextHandle context);

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    SpecializationInfoHandle get_specialization_info(const NodeIO& io) noexcept override;

    const void* get_push_constant(GraphRun& run, const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    vk::Extent3D extent;
    PushConstant pc;
    SpecializationInfoHandle spec_info;
    ShaderModuleHandle shader;

    ManagedVkImageInHandle con_src = ManagedVkImageIn::compute_read("src");
};

} // namespace merian_nodes
