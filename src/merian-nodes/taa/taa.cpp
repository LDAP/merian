#include "taa.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

static const uint32_t spv[] = {
#include "taa.comp.spv.h"
};

namespace merian {

TAANode::TAANode(const SharedContext context,
                 const ResourceAllocatorHandle allocator,
                 const float alpha,
                 const int clamp_method,
                 const bool inverse_motion)
    : ComputeNode(context, allocator, sizeof(PushConstant)), inverse_motion(inverse_motion) {
    shader = std::make_shared<ShaderModule>(context, sizeof(spv), spv);
    pc.temporal_alpha = alpha;
    pc.clamp_method = clamp_method;
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
    spec_builder.add_entry(local_size_x, local_size_y, int(inverse_motion));
    return spec_builder.build();
}

const void* TAANode::get_push_constant([[maybe_unused]] GraphRun& run) {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t> TAANode::get_group_count() const noexcept {
    return {(width + local_size_x - 1) / local_size_x, (height + local_size_y - 1) / local_size_y,
            1};
}

ShaderModuleHandle TAANode::get_shader_module() {
    return shader;
}

void TAANode::get_configuration(Configuration& config) {
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
}

} // namespace merian
