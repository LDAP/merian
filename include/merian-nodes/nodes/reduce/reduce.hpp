#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian_nodes {

class Reduce : public AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 32;
    static constexpr uint32_t local_size_y = 32;

  public:
    Reduce(const ContextHandle& context,
           const std::optional<vk::Format>& output_format = std::nullopt);

    ~Reduce();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const merian_nodes::NodeIO& io) const noexcept override;

    VulkanEntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& props) override;

  private:
    const std::optional<vk::Format> output_format;

    std::string source;

    std::string initial_value = "vec4(0)";
    std::string reduction = "accumulator + current_value";

    vk::Extent3D extent;
    VulkanEntryPointHandle shader;

    uint32_t number_inputs = 10;
    std::vector<VkSampledImageInHandle> input_connectors;
};

} // namespace merian_nodes
