#include "merian-nodes/nodes/taa/taa.hpp"

#include "config.h"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "taa.comp.spv.h"

namespace merian_nodes {

TAA::TAA(const ContextHandle& context) : AbstractCompute(context, sizeof(PushConstant)) {
    shader =
        std::make_shared<ShaderModule>(context, merian_taa_comp_spv_size(), merian_taa_comp_spv());
    pc.temporal_alpha = 0.;
    pc.clamp_method = MERIAN_NODES_TAA_CLAMP_MIN_MAX;

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, inverse_motion);
    spec_info = spec_builder.build();
}

std::vector<InputConnectorHandle> TAA::describe_inputs() {
    return {
        con_src,
        VkTextureIn::compute_read("prev_src", 1),
        con_mv,
    };
}

std::vector<OutputConnectorHandle>
TAA::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    width = io_layout[con_src]->create_info.extent.width;
    height = io_layout[con_src]->create_info.extent.height;

    pc.enable_mv = static_cast<VkBool32>(io_layout.is_connected(con_mv));
    return {
        ManagedVkImageOut::compute_write("out", io_layout[con_src]->create_info.format, width,
                                         height),
    };
}

SpecializationInfoHandle TAA::get_specialization_info([[maybe_unused]] const NodeIO& io) noexcept {
    return spec_info;
}

const void* TAA::get_push_constant([[maybe_unused]] GraphRun& run,
                                   [[maybe_unused]] const NodeIO& io) {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t>
TAA::get_group_count([[maybe_unused]] const NodeIO& io) const noexcept {
    return {(width + local_size_x - 1) / local_size_x, (height + local_size_y - 1) / local_size_y,
            1};
}

ShaderModuleHandle TAA::get_shader_module() {
    return shader;
}

AbstractCompute::NodeStatusFlags TAA::properties(Properties& config) {
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
