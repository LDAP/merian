#include "add.hpp"
#include "add.comp.spv.h"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

AddNode::AddNode(const SharedContext context,
                 const ResourceAllocatorHandle allocator,
                 const std::optional<vk::Format> output_format)
    : ComputeNode(context, allocator), output_format(output_format) {
    shader =
        std::make_shared<ShaderModule>(context, merian_add_comp_spv_size(), merian_add_comp_spv());
}

AddNode::~AddNode() {}

std::string AddNode::name() {
    return "Add";
}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
AddNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("a"),
            NodeInputDescriptorImage::compute_read("b"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
AddNode::describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                          const std::vector<NodeOutputDescriptorBuffer>&) {
    extent = connected_image_outputs[0].create_info.extent;
    vk::Format format = output_format.value_or(connected_image_outputs[0].create_info.format);

    return {
        {
            NodeOutputDescriptorImage::compute_write("output", format, extent),
        },
        {},
    };
}

SpecializationInfoHandle AddNode::get_specialization_info() const noexcept {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    return spec_builder.build();
}

// const void* AddNode::get_push_constant([[maybe_unused]] GraphRun& run) {
//     return &pc;
// }

std::tuple<uint32_t, uint32_t, uint32_t> AddNode::get_group_count() const noexcept {
    return {(extent.width + local_size_x - 1) / local_size_x,
            (extent.height + local_size_y - 1) / local_size_y, 1};
};

ShaderModuleHandle AddNode::get_shader_module() {
    return shader;
}

void AddNode::get_configuration(Configuration&, bool&) {}

} // namespace merian
