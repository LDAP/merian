#include "fxaa.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "fxaa.comp.spv.h"

namespace merian {

FXAA::FXAA(const SharedContext context, const ResourceAllocatorHandle allocator)
    : ComputeNode(context, allocator, sizeof(PushConstant)) {}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
FXAA::describe_inputs() {
    return {{NodeInputDescriptorImage::compute_read("in")}, {}};
}

std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
           std::vector<merian::NodeOutputDescriptorBuffer>>
FXAA::describe_outputs(
    const std::vector<merian::NodeOutputDescriptorImage>& connected_image_outputs,
    const std::vector<merian::NodeOutputDescriptorBuffer>&) {
    extent = connected_image_outputs[0].create_info.extent;

    return {{NodeOutputDescriptorImage::compute_write(
                "out", connected_image_outputs[0].create_info.format, extent)},
            {}};
}

SpecializationInfoHandle FXAA::get_specialization_info() const noexcept {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    return spec_builder.build();
}

const void* FXAA::get_push_constant([[maybe_unused]] GraphRun& run) {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t> FXAA::get_group_count() const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
}

ShaderModuleHandle FXAA::get_shader_module() {
    return std::make_shared<ShaderModule>(context, merian_fxaa_comp_spv_size(),
                                          merian_fxaa_comp_spv());
}

void FXAA::get_configuration(Configuration& config, bool&) {
    config.config_bool("enable", pc.enable, "");
}

} // namespace merian
