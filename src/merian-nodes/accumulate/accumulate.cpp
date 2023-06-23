#include "accumulate.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/shader_module.hpp"

namespace merian {

static const uint32_t spv[] = {
#include "accumulate.comp.spv.h"
};

AccumulateF32ImageNode::AccumulateF32ImageNode(const SharedContext context,
                                               const ResourceAllocatorHandle alloc)
    : context(context), alloc(alloc) {
    auto builder = DescriptorSetLayoutBuilder()
                       .add_binding_storage_image()  // in
                       .add_binding_storage_image(); // out
    layout = builder.build_layout(context);

    auto shader = std::make_shared<ShaderModule>(context, sizeof(spv), spv);
    auto pipe_layout =
        PipelineLayoutBuilder(context).add_descriptor_set_layout(layout).build_pipeline_layout();

    auto spec_builder = SpecializationInfoBuilder();
    spec_builder.add_entry(local_size_x, local_size_y);
    auto spec_info = spec_builder.build();

    pipe = std::make_shared<ComputePipeline>(pipe_layout, shader, spec_info);
}

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

    merian::NodeOutputDescriptorImage out_img = connected_image_outputs[0];
    out_img.name = "dst";
    out_img.persistent = true;

    return {{out_img}, {}};
}

void AccumulateF32ImageNode::cmd_build(
    const vk::CommandBuffer& cmd,
    const std::vector<std::vector<merian::ImageHandle>>& image_inputs,
    const std::vector<std::vector<merian::BufferHandle>>&,
    const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
    const std::vector<std::vector<merian::BufferHandle>>&) {

    // Since this is persistent there should only be exacly one image
    cmd.clearColorImage(*image_outputs[0][0], vk::ImageLayout::eGeneral, {},
                        all_levels_and_layers());

    sets.clear();
    in_textures.clear();
    out_textures.clear();

    uint32_t num_sets = image_outputs.size();

    pool = std::make_shared<merian::DescriptorPool>(layout, num_sets);
    vk::ImageViewCreateInfo create_image_view{
        {}, VK_NULL_HANDLE,         vk::ImageViewType::e2D, vk::Format::eR32G32B32A32Sfloat,
        {}, first_level_and_layer()};

    TextureHandle out_tex, in_tex;
    for (uint32_t i = 0; i < num_sets; i++) {
        auto set = std::make_shared<merian::DescriptorSet>(pool);
        sets.push_back(set);
        {
            // in
            create_image_view.image = *image_inputs[i][0];
            in_tex = alloc->createTexture(image_outputs[i][0], create_image_view);
            out_textures.push_back(in_tex);
        }
        {
            // out
            create_image_view.image = *image_outputs[i][0];
            out_tex = alloc->createTexture(image_outputs[i][0], create_image_view);
            out_textures.push_back(out_tex);
        }
        DescriptorSetUpdate(set)
            .write_descriptor_texture(0, in_tex)
            .write_descriptor_texture(1, out_tex)
            .update(context);
    }

    group_count_x = (image_inputs[0][0]->get_extent().width + local_size_x - 1) / local_size_x;
    group_count_y = (image_inputs[0][0]->get_extent().height + local_size_y - 1) / local_size_y;
}

void AccumulateF32ImageNode::cmd_process(const vk::CommandBuffer& cmd,
                                         GraphRun&,
                                         const uint32_t set_index,
                                         const std::vector<merian::ImageHandle>&,
                                         const std::vector<merian::BufferHandle>&,
                                         const std::vector<merian::ImageHandle>&,
                                         const std::vector<merian::BufferHandle>&) {

    pipe->bind(cmd);
    pipe->bind_descriptor_set(cmd, sets[set_index]);
    cmd.dispatch(group_count_x, group_count_y, 1);
}

} // namespace merian
