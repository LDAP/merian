#include "ab_compare.hpp"
#include "merian/vk/utils/blits.hpp"

namespace merian_nodes {

ABCompareNode::ABCompareNode(const std::string& name,
                             const std::optional<vk::Format> output_format,
                             const std::optional<vk::Extent2D> output_extent)
    : Node(name), output_format(output_format), output_extent(output_extent) {}

ABCompareNode::~ABCompareNode() {}

std::vector<InputConnectorHandle> ABCompareNode::describe_inputs() {
    return {img_in_a, img_in_b};
}

// --------------------------------------------------------------------------------

ABSplitNode::ABSplitNode(const std::optional<vk::Format> output_format,
                         const std::optional<vk::Extent2D> output_extent)
    : ABCompareNode("ABSplitNode", output_format, output_extent) {}

std::vector<OutputConnectorHandle>
ABSplitNode::describe_outputs(const ConnectorIOMap& output_for_input) {

    vk::Format format = output_format.has_value() ? output_format.value()
                                                  : output_for_input[img_in_a]->create_info.format;
    vk::Extent3D extent = output_extent.has_value()
                              ? vk::Extent3D(output_extent.value(), 1)
                              : output_for_input[img_in_a]->create_info.extent;

    img_out = VkImageOut::transfer_write("out", format, extent.width, extent.height);

    return {img_out};
}

void ABSplitNode::process([[maybe_unused]] GraphRun& run,
                          const vk::CommandBuffer& cmd,
                          [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                          const NodeIO& io) {
    const ImageHandle& a = io[img_in_a];
    const ImageHandle& b = io[img_in_b];
    const ImageHandle& result = io[img_out];

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

ABSideBySideNode::ABSideBySideNode(const std::optional<vk::Format> output_format,
                                   const std::optional<vk::Extent2D> output_extent)
    : ABCompareNode("ABSideBySideNode", output_format, output_extent) {}

std::vector<OutputConnectorHandle>
ABSideBySideNode::describe_outputs(const ConnectorIOMap& output_for_input) {

    vk::Format format = output_format.has_value() ? output_format.value()
                                                  : output_for_input[img_in_a]->create_info.format;

    vk::Extent3D extent;
    if (output_extent.has_value()) {
        extent = vk::Extent3D(output_extent.value(), 1);
    } else {
        extent = output_for_input[img_in_a]->create_info.extent;
        extent.width *= 2;
    }

    img_out = VkImageOut::transfer_write("out", format, extent.width, extent.height);

    return {img_out};
}

void ABSideBySideNode::process([[maybe_unused]] GraphRun& run,
                               [[maybe_unused]] const vk::CommandBuffer& cmd,
                               [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                               [[maybe_unused]] const NodeIO& io) {
    const ImageHandle& a = io[img_in_a];
    const ImageHandle& b = io[img_in_b];
    const ImageHandle& result = io[img_out];

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
