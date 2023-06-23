#include "ab_compare.hpp"

namespace merian {

ABCompareNode::ABCompareNode(const std::optional<vk::Format> output_format,
                             const std::optional<vk::Extent2D> output_extent)
    : output_format(output_format), output_extent(output_extent) {}

ABCompareNode::~ABCompareNode() {}

std::string ABCompareNode::name() {
    return "ABCompareNode";
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
ABCompareNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::transfer_src("a"),
            NodeInputDescriptorImage::transfer_src("b"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
ABCompareNode::describe_outputs(
    const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
    const std::vector<NodeOutputDescriptorBuffer>&) {

    vk::Format format = output_format.has_value() ? output_format.value()
                                                  : connected_image_outputs[0].create_info.format;
    vk::Extent3D extent = output_extent.has_value() ? vk::Extent3D(output_extent.value(), 1)
                                                    : connected_image_outputs[0].create_info.extent;

    return {
        {
            NodeOutputDescriptorImage::transfer_write("result", format, extent.width,
                                                      extent.height),
        },
        {},
    };
}

void ABCompareNode::cmd_process(const vk::CommandBuffer& cmd,
                                GraphRun&,
                                const uint32_t,
                                const std::vector<ImageHandle>& image_inputs,
                                const std::vector<BufferHandle>&,
                                const std::vector<ImageHandle>& image_outputs,
                                const std::vector<BufferHandle>&) {
    const ImageHandle& a = image_inputs[0];
    const ImageHandle& b = image_inputs[1];
    const ImageHandle& result = image_outputs[0];

    cmd_blit_fit(cmd, *b, vk::ImageLayout::eTransferSrcOptimal, b->get_extent(), *result,
                 vk::ImageLayout::eTransferDstOptimal, result->get_extent());

    vk::Extent3D a_extent = a->get_extent();
    a_extent.width /= 2;
    vk::Extent3D result_extent = result->get_extent();
    result_extent.width /= 2;

    cmd_blit_fit(cmd, *a, vk::ImageLayout::eTransferSrcOptimal, a_extent, *result,
                 vk::ImageLayout::eTransferDstOptimal, result_extent, {}, false);
}

} // namespace merian
