#include "merian-graph/nodes/ab_compare/ab_compare.hpp"
#include "merian/vk/utils/blits.hpp"

namespace merian {

AbstractABCompare::~AbstractABCompare() {}

std::vector<InputConnectorDescriptor> AbstractABCompare::describe_inputs() {
    return {{"a", con_in_a, ConnectorAccess::transfer_src},
            {"b", con_in_b, ConnectorAccess::transfer_src}};
}

std::vector<OutputConnectorDescriptor> ABSplit::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_in_a]->get_create_info_or_throw();

    vk::Format format = output_format.has_value() ? output_format.value() : create_info.format;
    vk::Extent3D extent =
        output_extent.has_value() ? vk::Extent3D(output_extent.value(), 1) : create_info.extent;

    con_out = ManagedVkImageOut::create(format, extent.width, extent.height);

    return {{"out", con_out, ConnectorAccess::transfer_dst}};
}

void ABSplit::process([[maybe_unused]] GraphRun& run, const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();
    const ImageHandle& a = io[con_in_a];
    const ImageHandle& b = io[con_in_b];
    const ImageHandle& result = io[con_out];

    cmd_blit_fit(cmd, b, vk::ImageLayout::eGeneral, b->get_extent(), result,
                 vk::ImageLayout::eGeneral, result->get_extent());

    cmd->barrier(result->barrier2(vk::ImageLayout::eGeneral));

    vk::Extent3D a_extent = a->get_extent();
    a_extent.width /= 2;
    vk::Extent3D result_extent = result->get_extent();
    result_extent.width /= 2;

    cmd_blit_fit(cmd, a, vk::ImageLayout::eGeneral, a_extent, result, vk::ImageLayout::eGeneral,
                 result_extent, std::nullopt);
}

// --------------------------------------------------------------------------------

std::vector<OutputConnectorDescriptor>
ABSideBySide::describe_outputs(const NodeIOLayout& io_layout) {
    const vk::ImageCreateInfo create_info = io_layout[con_in_a]->get_create_info_or_throw();

    vk::Format format = output_format.has_value() ? output_format.value() : create_info.format;

    vk::Extent3D extent;
    if (output_extent.has_value()) {
        extent = vk::Extent3D(output_extent.value(), 1);
    } else {
        extent = create_info.extent;
        extent.width *= 2;
    }

    con_out = ManagedVkImageOut::create(format, extent.width, extent.height);

    return {{"out", con_out, ConnectorAccess::transfer_dst}};
}

void ABSideBySide::process([[maybe_unused]] GraphRun& run, [[maybe_unused]] const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();
    const ImageHandle& a = io[con_in_a];
    const ImageHandle& b = io[con_in_b];
    const ImageHandle& result = io[con_out];

    vk::Extent3D half_result_extent = result->get_extent();
    half_result_extent.width /= 2;

    cmd_blit_fit(cmd, a, vk::ImageLayout::eGeneral, a->get_extent(), result,
                 vk::ImageLayout::eGeneral, half_result_extent, vk::ClearColorValue());

    // manual blit since we need offset by half result extent...
    vk::ImageBlit region{merian::first_layer(), {}, merian::first_layer(), {}};
    region.srcOffsets[1] = to_offset(b->get_extent());

    std::tie(region.dstOffsets[0], region.dstOffsets[1]) =
        fit(region.srcOffsets[0], region.srcOffsets[1], {(int32_t)half_result_extent.width, 0, 0},
            to_offset(result->get_extent()));

    cmd->blit(b, b->get_current_layout(), result, result->get_current_layout(), region);
}

} // namespace merian
