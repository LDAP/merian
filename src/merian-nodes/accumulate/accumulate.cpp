#include "accumulate.hpp"

namespace merian {

static const uint32_t spv[] = {
#include "accumulate.comp.spv.h"
};



std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
AccumulateF32ImageNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage(
                "src", vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead,
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage),
        },
        {},
    };
}

std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
           std::vector<merian::NodeOutputDescriptorBuffer>>
AccumulateF32ImageNode::describe_outputs(
    const std::vector<merian::NodeOutputDescriptorImage>& connected_image_outputs,
    const std::vector<merian::NodeOutputDescriptorBuffer>&) {

    assert(connected_image_outputs[0].create_info.imageType == vk::ImageType::e2D);
    assert(connected_image_outputs[0].create_info.mipLevels == 1);
    assert(connected_image_outputs[0].create_info.arrayLayers == 1);
    assert(connected_image_outputs[0].create_info.format == vk::Format::eR32G32B32A32Sfloat);


    vk::ImageCreateInfo create_image{{},
                                     vk::ImageType::e2D,
                                     connected_image_outputs[0].create_info.format,
                                     connected_image_outputs[0].create_info.extent,
                                     1,
                                     1,
                                     vk::SampleCountFlagBits::e1,
                                     vk::ImageTiling::eOptimal,
                                     vk::ImageUsageFlagBits::eStorage |
                                         vk::ImageUsageFlagBits::eTransferDst,
                                     vk::SharingMode::eExclusive,
                                     {},
                                     {},
                                     vk::ImageLayout::eUndefined};
    return {
        {
            merian::NodeOutputDescriptorImage{
                "result", vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
                create_image, vk::ImageLayout::eGeneral, false},
        },
        {},
    };
}
} // namespace merian
