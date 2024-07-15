#pragma once

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian_nodes {

class Add : public AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 32;
    static constexpr uint32_t local_size_y = 32;

  public:
    Add(const SharedContext context, const std::optional<vk::Format> output_format = std::nullopt);

    ~Add();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    SpecializationInfoHandle get_specialization_info(const NodeIO& io) noexcept override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const merian_nodes::NodeIO& io) const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    const std::optional<vk::Format> output_format;
    vk::Extent3D extent;
    ShaderModuleHandle shader;
    SpecializationInfoHandle spec_info;

    static constexpr uint32_t number_inputs = 10;
    std::vector<ManagedVkImageInHandle> input_connectors;
};

} // namespace merian_nodes
