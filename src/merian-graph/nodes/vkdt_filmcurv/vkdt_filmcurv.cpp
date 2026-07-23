#include "merian-graph/nodes/vkdt_filmcurv/vkdt_filmcurv.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

namespace {
constexpr const char* SHADER_MODULE = "merian-graph/nodes/vkdt_filmcurv/vkdt_filmcurv.slang";
}

VKDTFilmcurv::VKDTFilmcurv() : TypedPCAbstractCompute() {}

DeviceSupportInfo VKDTFilmcurv::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(SHADER_MODULE, true);
    return SlangProgram::create(query_info.compile_context, composition)
        .get()
        ->query_device_support(query_info);
}

void VKDTFilmcurv::initialize(const ContextHandle& context,
                              const ResourceAllocatorHandle& allocator) {
    TypedPCAbstractCompute::initialize(context, allocator);

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    spec_info.set(spec_builder.build());
}

SlangCompositionHandle VKDTFilmcurv::create_composition() {
    const auto composition = SlangComposition::create();
    composition->add_module_from_path(SHADER_MODULE, true);
    return composition;
}

std::vector<InputConnectorDescriptor> VKDTFilmcurv::describe_inputs() {
    return {
        {"src", con_src, ConnectorAccess::compute_read},
    };
}

std::vector<OutputConnectorDescriptor>
VKDTFilmcurv::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_src]->get_create_info_or_throw();

    extent = create_info.extent;
    const vk::Format format = output_format.value_or(create_info.format);

    return {
        {"out", ManagedVkImageOut::create(format, extent), ConnectorAccess::compute_write},
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

AbstractCompute::NodeStatusFlags VKDTFilmcurv::properties(Properties& config) {
    config.config_float("brightness", pc.brightness, "", .01);
    config.config_float("contrast", pc.contrast, "", .01);
    config.config_float("bias", pc.bias, "", .01);
    config.config_options("colormode", pc.colourmode,
                          {"darktable ucs", "per channel", "munsell", "hsl"});

    return {};
}

} // namespace merian
