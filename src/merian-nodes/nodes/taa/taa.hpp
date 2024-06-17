#pragma once

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"
#include "merian-nodes/nodes/taa/config.h"

namespace merian_nodes {

class TAA : public AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        // higher value means more temporal reuse
        float temporal_alpha;
        int clamp_method;
    };

  public:
    TAA(const SharedContext context,
        const float alpha = 0.,
        const int clamp_method = MERIAN_NODES_TAA_CLAMP_MIN_MAX,
        const bool inverse_motion = false);

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    SpecializationInfoHandle get_specialization_info() const noexcept override;

    const void* get_push_constant([[maybe_unused]] GraphRun& run) override;

    std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    const bool inverse_motion;
    ShaderModuleHandle shader;
    SpecializationInfoHandle spec_info;

    ManagedVkImageInHandle con_src = ManagedVkImageIn::compute_read("src");

    PushConstant pc;
    uint32_t width{};
    uint32_t height{};
};

} // namespace merian_nodes
