#include "merian-nodes/nodes/fxaa/fxaa.hpp"

#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "fxaa.comp.spv.h"

namespace merian_nodes {

FXAA::FXAA(const ContextHandle& context) : AbstractCompute(context, sizeof(PushConstant)) {
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
FXAA::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    extent = io_layout[con_src]->create_info.extent;

    return {
        ManagedVkImageOut::compute_write("out", io_layout[con_src]->create_info.format, extent),
    };
}

SpecializationInfoHandle FXAA::get_specialization_info([[maybe_unused]] const NodeIO& io) noexcept {
    return spec_info;
}

const void* FXAA::get_push_constant([[maybe_unused]] GraphRun& run,
                                    [[maybe_unused]] const NodeIO& io) {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t>
FXAA::get_group_count([[maybe_unused]] const NodeIO& io) const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
}

ShaderModuleHandle FXAA::get_shader_module() {
    return shader;
}

AbstractCompute::NodeStatusFlags FXAA::properties(Properties& config) {
    config.config_bool("enable", pc.enable, "");
    config.config_float("subpixel alias removal", pc.fxaaQualitySubpix, 0., 1.,
                        R"(Choose the amount of sub-pixel aliasing removal.
This can effect sharpness.
  1.00 - upper limit (softer)
  0.75 - default amount of filtering
  0.50 - lower limit (sharper, less sub-pixel aliasing removal)
  0.25 - almost off
  0.00 - completely off
)");
    config.config_float("edge threshold", pc.fxaaQualityEdgeThreshold, 0.063, 0.333,
                        R"(The minimum amount of local contrast required to apply algorithm.
  0.333 - too little (faster)
  0.250 - low quality
  0.166 - default
  0.125 - high quality 
  0.063 - overkill (slower)
)");
    config.config_float("edge threshold min", pc.fxaaQualityEdgeThresholdMin, 0.01, 0.1,
                        R"(Trims the algorithm from processing darks.
  0.0833 - upper limit (default, the start of visible unfiltered edges)
  0.0625 - high quality (faster)
  0.0312 - visible limit (slower)
)");

    return {};
}

} // namespace merian_nodes
