#include "accumulate.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"

static const uint32_t spv[] = {
#include "accumulate.comp.spv.h"
};

namespace merian {

AccumulateNode::AccumulateNode(const SharedContext context, const ResourceAllocatorHandle allocator)
    : ComputeNode(context, allocator, sizeof(AccumulatePushConstant)) {
    shader = std::make_shared<ShaderModule>(context, sizeof(spv), spv);
}

AccumulateNode::~AccumulateNode() {}

std::string AccumulateNode::name() {
    return "Accumulate";
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
AccumulateNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("prev_accum", 1),
            NodeInputDescriptorImage::compute_read("prev_moments", 1),

            NodeInputDescriptorImage::compute_read("irr"),
            NodeInputDescriptorImage::compute_read("gbuf"),
            NodeInputDescriptorImage::compute_read("prev_gbuf", 1),

            NodeInputDescriptorImage::compute_read("mv"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
AccumulateNode::describe_outputs(
    const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
    const std::vector<NodeOutputDescriptorBuffer>&) {
    extent = connected_image_outputs[2].create_info.extent;

    // clang-format off
    return {
        {
            NodeOutputDescriptorImage::compute_read_write("accum", vk::Format::eR32G32B32A32Sfloat, extent),
            NodeOutputDescriptorImage::compute_read_write("moments", vk::Format::eR32G32B32A32Sfloat, extent),
        },
        {},
    };
    // clang-format on
}

SpecializationInfoHandle AccumulateNode::get_specialization_info() const noexcept {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    return spec_builder.build();
}

const void* AccumulateNode::get_push_constant() {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t> AccumulateNode::get_group_count() const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle AccumulateNode::get_shader_module() {
    return shader;
}

void AccumulateNode::get_configuration(Configuration& config) {
    config.config_float("alpha", pc.accum_alpha, 0, 1,
                        "Blend factor with the previous information. More means more reuse");
    config.config_angle("normal threshold", pc.normal_reject_rad,
                        "Reject points with normals farther apart", 0, 90);
    config.config_percent("depth threshold", pc.depth_reject_percent,
                          "Reject points with depths farther apart (relative to the max)");
    config.config_options("filter mode", pc.filter_mode, {"nearest", "linear"});
}

} // namespace merian
