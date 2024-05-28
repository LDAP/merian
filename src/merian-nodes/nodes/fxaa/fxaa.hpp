#pragma once

#include "merian-nodes/nodes/compute_node/compute_node.hpp"
#include "merian-nodes/connectors/vk_image_in.hpp"

namespace merian_nodes {

class FXAA : public AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

    struct PushConstant {
        int32_t enable = 1;
    };

  public:
    FXAA(const SharedContext context);

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    SpecializationInfoHandle get_specialization_info() const noexcept override;

    const void* get_push_constant(GraphRun& run) override;

    std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags configuration(Configuration& config) override;

  private:
    vk::Extent3D extent;
    PushConstant pc;
    SpecializationInfoHandle spec_info;
    ShaderModuleHandle shader;

    VkImageInHandle con_src = VkImageIn::compute_read("src");
};

} // namespace merian_nodes
