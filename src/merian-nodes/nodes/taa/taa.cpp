#include "merian-nodes/nodes/taa/taa.hpp"

#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "config.h"
#include "taa.slang.spv.h"

namespace merian_nodes {

TAA::TAA(const ContextHandle& context) : AbstractCompute(context, sizeof(PushConstant)) {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, inverse_motion);
    spec_info = spec_builder.build();

    shader = EntryPoint::create(context, merian_taa_slang_spv(), merian_taa_slang_spv_size(),
                                "main", vk::ShaderStageFlagBits::eCompute, spec_info);
    pc.temporal_alpha = 0.;
    pc.clamp_method = MERIAN_NODES_TAA_CLAMP_MIN_MAX;
}

std::vector<InputConnectorHandle> TAA::describe_inputs() {
    return {
        con_src,
        VkSampledImageIn::compute_read("prev_src", 1),
        con_mv,
    };
}

std::vector<OutputConnectorHandle>
TAA::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();
    width = create_info.extent.width;
    height = create_info.extent.height;

    pc.enable_mv = static_cast<VkBool32>(io_layout.is_connected(con_mv));
    return {
        ManagedVkImageOut::compute_write("out", create_info.format, width, height),
    };
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

SpecializedEntryPointHandle TAA::get_entry_point() {
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
