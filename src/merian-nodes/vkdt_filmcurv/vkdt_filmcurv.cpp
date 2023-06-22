#include "vkdt_filmcurv.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

static const uint32_t spv[] = {
#include "vkdt_filmcurv.comp.spv.h"
};

namespace merian {

VKDTFilmcurv::VKDTFilmcurv(const SharedContext context,
                           const ResourceAllocatorHandle allocator,
                           const vk::Format output_format)
    : ComputeNode(context, allocator, sizeof(PushConstant)), output_format(output_format) {}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
VKDTFilmcurv::describe_inputs() {
    return {{NodeInputDescriptorImage::compute_read("in")}, {}};
}

std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
           std::vector<merian::NodeOutputDescriptorBuffer>>
VKDTFilmcurv::describe_outputs(
    const std::vector<merian::NodeOutputDescriptorImage>& connected_image_outputs,
    const std::vector<merian::NodeOutputDescriptorBuffer>&) {
    const vk::Extent3D& extent = connected_image_outputs[0].create_info.extent;

    width = extent.width;
    height = extent.height;

    return {{NodeOutputDescriptorImage::compute_write("out", output_format, width, height)}, {}};
}

SpecializationInfoHandle VKDTFilmcurv::get_specialization_info() const noexcept {
    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    return spec_builder.build();
}

const void* VKDTFilmcurv::get_push_constant() {
    return &pc;
}

std::tuple<uint32_t, uint32_t, uint32_t> VKDTFilmcurv::get_group_count() const noexcept {
    return {(width + local_size_x - 1) / local_size_x, (height + local_size_y - 1) / local_size_y,
            1};
}

ShaderModuleHandle VKDTFilmcurv::get_shader_module() {
    return std::make_shared<ShaderModule>(context, sizeof(spv), spv);
}

} // namespace merian
