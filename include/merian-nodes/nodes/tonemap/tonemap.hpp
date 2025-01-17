#pragma once

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/nodes/compute_node/compute_node.hpp"

namespace merian_nodes {

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
    Tonemap(const ContextHandle& context,
            const std::optional<vk::Format> output_format = std::nullopt);

    ~Tonemap();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    SpecializationInfoHandle get_specialization_info(const NodeIO& io) noexcept override;

    const void* get_push_constant(GraphRun& run, const NodeIO& io) override;

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count(const NodeIO& io) const noexcept override;

    ShaderModuleHandle get_shader_module() override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    void make_spec_info();

    const std::optional<vk::Format> output_format;

    ManagedVkImageInHandle con_src = ManagedVkImageIn::compute_read("src");

    vk::Extent3D extent;
    PushConstant pc;
    ShaderModuleHandle shader;
    SpecializationInfoHandle spec_info;

    int32_t tonemap = 0;
    int32_t alpha_mode = 0;
    int32_t clamp_output = 1;
};

} // namespace merian_nodes
