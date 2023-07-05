#pragma once

#include "taa.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

static const uint32_t spv[] = {
#include "taa.comp.spv.h"
};

namespace merian {

TAANode::TAANode(const SharedContext context,
                 const ResourceAllocatorHandle allocator,
                 const int clamp_method,
                 const bool inverse_motion)
    : ComputeNode(context, allocator, sizeof(PushConstant)), clamp_method(clamp_method),
      inverse_motion(inverse_motion) {
    shader = std::make_shared<ShaderModule>(context, sizeof(spv), spv);
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
TAANode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("current"),
            NodeInputDescriptorImage::compute_read("previous", 1),
            NodeInputDescriptorImage::compute_read("mv"),
        },
        {},
    };
};

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
TAANode::describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                          const std::vector<NodeOutputDescriptorBuffer>&) {
    const NodeOutputDescriptorImage& current_input = connected_image_outputs[0];
    width = current_input.create_info.extent.width;
    height = current_input.create_info.extent.height;

    return {
        {
            merian::NodeOutputDescriptorImage::compute_write(
                "out", current_input.create_info.format, width, height),
        },
        {},
    };
};

SpecializationInfoHandle TAANode::get_specialization_info() const noexcept {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y, clamp_method, int(inverse_motion));
    return spec_builder.build();
}

const void* TAANode::get_push_constant() {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t> TAANode::get_group_count() const noexcept {
    return {(width + local_size_x - 1) / local_size_x, (height + local_size_y - 1) / local_size_y,
            1};
}

ShaderModuleHandle TAANode::get_shader_module() {
    return shader;
}

} // namespace merian
