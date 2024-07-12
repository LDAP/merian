#include "vkdt_filmcurv.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

static const uint32_t spv[] = {
#include "vkdt_filmcurv.comp.spv.h"
};

namespace merian_nodes {

VKDTFilmcurv::VKDTFilmcurv(const SharedContext context,
                           const std::optional<Options> options,
                           const std::optional<vk::Format> output_format)
    : AbstractCompute(context, sizeof(Options)), output_format(output_format) {
    if (options) {
        pc = options.value();
    }

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info = spec_builder.build();

    shader = std::make_shared<ShaderModule>(context, sizeof(spv), spv);
}

std::vector<InputConnectorHandle> VKDTFilmcurv::describe_inputs() {
    return {
        con_src,
    };
}

std::vector<OutputConnectorHandle>
VKDTFilmcurv::describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
    extent = output_for_input[con_src]->create_info.extent;
    const vk::Format format = output_format.value_or(output_for_input[con_src]->create_info.format);

    return {
        ManagedVkImageOut::compute_write("out", format, extent),
    };
}

SpecializationInfoHandle
VKDTFilmcurv::get_specialization_info([[maybe_unused]] const NodeIO& io) const noexcept {
    return spec_info;
}

const void* VKDTFilmcurv::get_push_constant([[maybe_unused]] GraphRun& run,
                                            [[maybe_unused]] const NodeIO& io) {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t>
VKDTFilmcurv::get_group_count([[maybe_unused]] const NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
}

ShaderModuleHandle VKDTFilmcurv::get_shader_module() {
    return shader;
}

AbstractCompute::NodeStatusFlags VKDTFilmcurv::properties(Properties& config) {
    config.config_float("brightness", pc.brightness, "", .01);
    config.config_float("contrast", pc.contrast, "", .01);
    config.config_float("bias", pc.bias, "", .01);
    config.config_options("colormode", pc.colourmode,
                          {"darktable ucs", "per channel", "munsell", "hsl"});

    return {};
}

} // namespace merian_nodes
