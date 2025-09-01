#pragma once

#include "merian-nodes/connectors/image/vk_image_in_sampled.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian_nodes {

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
    VKDTFilmcurv(const ContextHandle& context,
                 const std::optional<VKDTFilmcurvePushConstant> options = std::nullopt,
                 const std::optional<vk::Format> output_format = std::nullopt);

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    const VKDTFilmcurvePushConstant& get_typed_push_constant(GraphRun& run,
                                                             const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    EntryPointHandle get_entry_point() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    const std::optional<vk::Format> output_format;

    VkSampledImageInHandle con_src = VkSampledImageIn::compute_read("src");
    vk::Extent3D extent;

    EntryPointHandle shader;
    SpecializationInfoHandle spec_info;

    VKDTFilmcurvePushConstant pc;
};

} // namespace merian_nodes
