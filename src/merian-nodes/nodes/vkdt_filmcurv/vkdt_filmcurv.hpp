#pragma once

#include "merian-nodes/nodes/compute_node/compute_node.hpp"
#include "merian-nodes/connectors/vk_image_in.hpp"

namespace merian_nodes {

class VKDTFilmcurv : public AbstractCompute {

  private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;

  public:
    struct Options {
        float brightness{1.};
        float contrast{1.};
        float bias{0.};
        int32_t colourmode{1};
    };

  public:
    VKDTFilmcurv(const SharedContext context,
                 const std::optional<Options> options = std::nullopt,
                 const std::optional<vk::Format> output_format = std::nullopt);

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    SpecializationInfoHandle get_specialization_info() const noexcept override;

    const void* get_push_constant([[maybe_unused]] GraphRun& run) override;

    std::tuple<uint32_t, uint32_t, uint32_t> get_group_count() const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags configuration(Configuration& config) override;

  private:
    const std::optional<vk::Format> output_format;

    VkImageInHandle con_src = VkImageIn::compute_read("src");
    vk::Extent3D extent;

    ShaderModuleHandle shader;
    SpecializationInfoHandle spec_info;

    Options pc;
};

} // namespace merian
