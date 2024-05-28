#include "taa.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

static const uint32_t spv[] = {
#include "taa.comp.spv.h"
};

namespace merian_nodes {

TAA::TAA(const SharedContext context,
                 const float alpha,
                 const int clamp_method,
                 const bool inverse_motion)
    : AbstractCompute(context, "Temporal Anti-Aliasing", sizeof(PushConstant)),
      inverse_motion(inverse_motion) {
    shader = std::make_shared<ShaderModule>(context, sizeof(spv), spv);
    pc.temporal_alpha = alpha;
    pc.clamp_method = clamp_method;

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, int(inverse_motion));
    spec_info = spec_builder.build();
}

std::vector<InputConnectorHandle> TAA::describe_inputs() {
    return {
        con_src,
        VkImageIn::compute_read("feedback", 1),
        VkImageIn::compute_read("mv"),
    };
}

std::vector<OutputConnectorHandle>
TAA::describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
    width = output_for_input[con_src]->create_info.extent.width;
    height = output_for_input[con_src]->create_info.extent.height;

    return {
        VkImageOut::compute_write("out", output_for_input[con_src]->create_info.format, width,
                                  height),
    };
}

SpecializationInfoHandle TAA::get_specialization_info() const noexcept {
    return spec_info;
}

const void* TAA::get_push_constant([[maybe_unused]] GraphRun& run) {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t> TAA::get_group_count() const noexcept {
    return {(width + local_size_x - 1) / local_size_x, (height + local_size_y - 1) / local_size_y,
            1};
}

ShaderModuleHandle TAA::get_shader_module() {
    return shader;
}

AbstractCompute::NodeStatusFlags TAA::configuration(Configuration& config) {
    config.config_percent("alpha", pc.temporal_alpha, "more means more reuse");

    std::vector<std::string> clamp_methods = {
        fmt::format("none ({})", MERIAN_NODES_TAA_CLAMP_NONE),
        fmt::format("min-max ({})", MERIAN_NODES_TAA_CLAMP_MIN_MAX),
        fmt::format("moments ({})", MERIAN_NODES_TAA_CLAMP_MOMENTS),
    };
    config.config_options("clamp method", pc.clamp_method, clamp_methods);

    std::string text;
    text += fmt::format("inverse motion: {}", inverse_motion);
    config.output_text(text);

    return {};
}

} // namespace merian_nodes
