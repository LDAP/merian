#include "merian-graph/nodes/taa/taa.hpp"

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/nodes/taa/config.h"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

namespace {
constexpr const char* SHADER_MODULE = "merian-graph/nodes/taa/taa.slang";
}

DeviceSupportInfo TAA::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(SHADER_MODULE, true);
    return SlangProgram::create(query_info.compile_context, composition)
        .get()
        ->query_device_support(query_info);
}

TAA::TAA() : AbstractCompute(sizeof(PushConstant)) {
    pc.temporal_alpha = 0.;
    pc.clamp_method = MERIAN_NODES_TAA_CLAMP_MIN_MAX;
}

void TAA::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    AbstractCompute::initialize(context, allocator);

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, inverse_motion);
    spec_info.set(spec_builder.build());
}

SlangCompositionHandle TAA::create_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(SHADER_MODULE, true);
    return composition;
}

std::vector<InputConnectorDescriptor> TAA::describe_inputs() {
    return {
        {"src", con_src, ConnectorAccess::compute_read},
        {"prev_src", VkSampledImageIn::create(), ConnectorAccess::compute_read, 1},
        {"mv", con_mv, ConnectorAccess::compute_read, 0, true},
    };
}

std::vector<OutputConnectorDescriptor>
TAA::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();
    width = create_info.extent.width;
    height = create_info.extent.height;

    pc.enable_mv = static_cast<VkBool32>(io_layout.is_connected(con_mv));
    return {
        {"out", ManagedVkImageOut::create(create_info.format, width, height),
         ConnectorAccess::compute_write},
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

} // namespace merian
