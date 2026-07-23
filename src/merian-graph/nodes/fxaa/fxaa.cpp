#include "merian-graph/nodes/fxaa/fxaa.hpp"

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

namespace {
constexpr const char* SHADER_MODULE = "merian-graph/nodes/fxaa/fxaa.slang";
}

FXAA::FXAA() : AbstractCompute(sizeof(PushConstant)) {}

DeviceSupportInfo FXAA::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(SHADER_MODULE, true);
    return SlangProgram::create(query_info.compile_context, composition)
        .get()
        ->query_device_support(query_info);
}

void FXAA::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    AbstractCompute::initialize(context, allocator);

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info.set(spec_builder.build());
}

SlangCompositionHandle FXAA::create_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(SHADER_MODULE, true);
    return composition;
}

std::vector<InputConnectorDescriptor> FXAA::describe_inputs() {
    return {
        {"src", con_src, ConnectorAccess::compute_read},
    };
}

std::vector<OutputConnectorDescriptor>
FXAA::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();

    extent = create_info.extent;

    return {
        {"out", ManagedVkImageOut::create(create_info.format, extent),
         ConnectorAccess::compute_write},
    };
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

AbstractCompute::NodeStatusFlags FXAA::properties(Properties& config) {
    config.config_bool("enable", pc.enable, "");
    config.config_float("subpixel alias removal", pc.fxaaQualitySubpix,
                        R"(Choose the amount of sub-pixel aliasing removal.
This can effect sharpness.
  1.00 - upper limit (softer)
  0.75 - default amount of filtering
  0.50 - lower limit (sharper, less sub-pixel aliasing removal)
  0.25 - almost off
  0.00 - completely off
)",
                        0.01f, 0.0f, 1.0f);
    config.config_float("edge threshold", pc.fxaaQualityEdgeThreshold,
                        R"(The minimum amount of local contrast required to apply algorithm.
  0.333 - too little (faster)
  0.250 - low quality
  0.166 - default
  0.125 - high quality
  0.063 - overkill (slower)
)",
                        0.001f, 0.063f, 0.333f);
    config.config_float("edge threshold min", pc.fxaaQualityEdgeThresholdMin,
                        R"(Trims the algorithm from processing darks.
  0.0833 - upper limit (default, the start of visible unfiltered edges)
  0.0625 - high quality (faster)
  0.0312 - visible limit (slower)
)",
                        0.001f, 0.01f, 0.1f);

    return {};
}

} // namespace merian
