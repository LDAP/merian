#include "ab_compare.hpp"
#include "merian/vk/utils/blits.hpp"

namespace merian_nodes {

AbstractABCompare::AbstractABCompare(const std::optional<vk::Format> output_format,
                                     const std::optional<vk::Extent2D> output_extent)
    : Node(), output_format(output_format), output_extent(output_extent) {}

AbstractABCompare::~AbstractABCompare() {}

std::vector<InputConnectorHandle> AbstractABCompare::describe_inputs() {
    return {con_in_a, con_in_b};
}

// --------------------------------------------------------------------------------

ABSplit::ABSplit(const std::optional<vk::Format> output_format,
                 const std::optional<vk::Extent2D> output_extent)
    : AbstractABCompare(output_format, output_extent) {}

std::vector<OutputConnectorHandle> ABSplit::describe_outputs(const NodeIOLayout& io_layout) {

    vk::Format format =
        output_format.has_value() ? output_format.value() : io_layout[con_in_a]->create_info.format;
    vk::Extent3D extent = output_extent.has_value() ? vk::Extent3D(output_extent.value(), 1)
                                                    : io_layout[con_in_a]->create_info.extent;

    con_out = ManagedVkImageOut::transfer_write("out", format, extent.width, extent.height);

    return {con_out};
}

void ABSplit::process([[maybe_unused]] GraphRun& run,
                      const vk::CommandBuffer& cmd,
                      [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                      const NodeIO& io) {
    const ImageHandle& a = io[con_in_a];
    const ImageHandle& b = io[con_in_b];
    const ImageHandle& result = io[con_out];

    cmd_blit_fit(cmd, *b, vk::ImageLayout::eTransferSrcOptimal, b->get_extent(), *result,
                 vk::ImageLayout::eTransferDstOptimal, result->get_extent());

    vk::Extent3D a_extent = a->get_extent();
    a_extent.width /= 2;
    vk::Extent3D result_extent = result->get_extent();
    result_extent.width /= 2;

    cmd_blit_fit(cmd, *a, vk::ImageLayout::eTransferSrcOptimal, a_extent, *result,
                 vk::ImageLayout::eTransferDstOptimal, result_extent, std::nullopt);
}

// --------------------------------------------------------------------------------

ABSideBySide::ABSideBySide(const std::optional<vk::Format> output_format,
                           const std::optional<vk::Extent2D> output_extent)
    : AbstractABCompare(output_format, output_extent) {}

std::vector<OutputConnectorHandle> ABSideBySide::describe_outputs(const NodeIOLayout& io_layout) {

    vk::Format format =
        output_format.has_value() ? output_format.value() : io_layout[con_in_a]->create_info.format;

    vk::Extent3D extent;
    if (output_extent.has_value()) {
        extent = vk::Extent3D(output_extent.value(), 1);
    } else {
        extent = io_layout[con_in_a]->create_info.extent;
        extent.width *= 2;
    }

    con_out = ManagedVkImageOut::transfer_write("out", format, extent.width, extent.height);

    return {con_out};
}

void ABSideBySide::process([[maybe_unused]] GraphRun& run,
                           [[maybe_unused]] const vk::CommandBuffer& cmd,
                           [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                           [[maybe_unused]] const NodeIO& io) {
    const ImageHandle& a = io[con_in_a];
    const ImageHandle& b = io[con_in_b];
    const ImageHandle& result = io[con_out];

    vk::Extent3D half_result_extent = result->get_extent();
    half_result_extent.width /= 2;

    cmd_blit_fit(cmd, *a, vk::ImageLayout::eTransferSrcOptimal, a->get_extent(), *result,
                 vk::ImageLayout::eTransferDstOptimal, half_result_extent, vk::ClearColorValue());

    // manual blit since we need offset by half result extent...
    vk::ImageBlit region{merian::first_layer(), {}, merian::first_layer(), {}};
    region.srcOffsets[1] = extent_to_offset(b->get_extent());

    std::tie(region.dstOffsets[0], region.dstOffsets[1]) =
        fit(region.srcOffsets[0], region.srcOffsets[1], {(int32_t)half_result_extent.width, 0, 0},
            extent_to_offset(result->get_extent()));

    cmd.blitImage(*b, b->get_current_layout(), *result, result->get_current_layout(), {region},
                  vk::Filter::eLinear);
}

} // namespace merian_nodes
