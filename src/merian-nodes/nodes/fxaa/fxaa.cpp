#include "fxaa.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "fxaa.comp.spv.h"

namespace merian_nodes {

FXAA::FXAA(const SharedContext context) : AbstractCompute(context, "FXAA", sizeof(PushConstant)) {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info = spec_builder.build();
    shader = std::make_shared<ShaderModule>(context, merian_fxaa_comp_spv_size(),
                                            merian_fxaa_comp_spv());
}

std::vector<InputConnectorHandle> FXAA::describe_inputs() {
    return {
        con_src,
    };
}

std::vector<OutputConnectorHandle>
FXAA::describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
    extent = output_for_input[con_src]->create_info.extent;

    return {
        ManagedVkImageOut::compute_write("out", output_for_input[con_src]->create_info.format, extent),
    };
}

SpecializationInfoHandle FXAA::get_specialization_info() const noexcept {
    return spec_info;
}

const void* FXAA::get_push_constant([[maybe_unused]] GraphRun& run) {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t> FXAA::get_group_count() const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
}

ShaderModuleHandle FXAA::get_shader_module() {
    return shader;
}

AbstractCompute::NodeStatusFlags FXAA::properties(Properties& config) {
    config.config_bool("enable", pc.enable, "");
    return {};
}

} // namespace merian_nodes
