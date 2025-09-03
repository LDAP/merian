#include "merian-nodes/nodes/vkdt_filmcurv/vkdt_filmcurv.hpp"
#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "vkdt_filmcurv.slang.spv.h"

namespace merian_nodes {

VKDTFilmcurv::VKDTFilmcurv(const ContextHandle& context,
                           const std::optional<VKDTFilmcurvePushConstant> options,
                           const std::optional<vk::Format> output_format)
    : TypedPCAbstractCompute(context), output_format(output_format) {
    if (options) {
        pc = options.value();
    }

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info = spec_builder.build();

    shader = EntryPoint::create(context, merian_vkdt_filmcurv_slang_spv(),
                                merian_vkdt_filmcurv_slang_spv_size(), "main",
                                vk::ShaderStageFlagBits::eCompute, spec_info);
}

std::vector<InputConnectorHandle> VKDTFilmcurv::describe_inputs() {
    return {
        con_src,
    };
}

std::vector<OutputConnectorHandle>
VKDTFilmcurv::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();

    extent = create_info.extent;
    const vk::Format format = output_format.value_or(create_info.format);

    return {
        ManagedVkImageOut::compute_write("out", format, extent),
    };
}

const VKDTFilmcurvePushConstant&
VKDTFilmcurv::get_typed_push_constant([[maybe_unused]] GraphRun& run,
                                      [[maybe_unused]] const NodeIO& io) {
    return pc;
}

std::tuple<uint32_t, uint32_t, uint32_t>
VKDTFilmcurv::get_group_count([[maybe_unused]] const NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
}

VulkanEntryPointHandle VKDTFilmcurv::get_entry_point() {
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
