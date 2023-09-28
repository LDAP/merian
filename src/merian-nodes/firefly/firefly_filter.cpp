#include "firefly_filter.hpp"

#include "firefly_filter.comp.spv.h"
#include "firefly_filter_quartile.comp.spv.h"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

FireflyFilterNode::FireflyFilterNode(const SharedContext context,
                                     const ResourceAllocatorHandle allocator)
    : context(context), allocator(allocator) {
    quartile_module =
        std::make_shared<ShaderModule>(context, merian_firefly_filter_quartile_comp_spv_size(),
                                       merian_firefly_filter_quartile_comp_spv());
    filter_module = std::make_shared<ShaderModule>(context, merian_firefly_filter_comp_spv_size(),
                                                   merian_firefly_filter_comp_spv());
}

FireflyFilterNode::~FireflyFilterNode() {}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
FireflyFilterNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("irr"),
            NodeInputDescriptorImage::compute_read("moments"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
FireflyFilterNode::describe_outputs(
    const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
    const std::vector<NodeOutputDescriptorBuffer>&) {

    irr_create_info = connected_image_outputs[0].create_info;
    const auto moments_create_info = connected_image_outputs[0].create_info;

    return {
        {
            NodeOutputDescriptorImage::compute_write("out_irr", irr_create_info.format,
                                                     irr_create_info.extent),
            NodeOutputDescriptorImage::compute_write("out_moments", moments_create_info.format,
                                                     moments_create_info.extent),
        },
        {},
    };
}

void FireflyFilterNode::cmd_build([[maybe_unused]] const vk::CommandBuffer& cmd,
                                  const std::vector<std::vector<ImageHandle>>& image_inputs,
                                  const std::vector<std::vector<BufferHandle>>& buffer_inputs,
                                  const std::vector<std::vector<ImageHandle>>& image_outputs,
                                  const std::vector<std::vector<BufferHandle>>& buffer_outputs) {
    std::tie(graph_textures, graph_sets, graph_pool, graph_layout) =
        make_graph_descriptor_sets(context, allocator, image_inputs, buffer_inputs, image_outputs,
                                   buffer_outputs, graph_layout);

    if (!quartile_desc_layout) {
        quartile_desc_layout =
            DescriptorSetLayoutBuilder().add_binding_storage_image().build_layout(context);
        filter_desc_layout =
            DescriptorSetLayoutBuilder().add_binding_combined_sampler().build_layout(context);

        quartile_desc_pool = std::make_shared<DescriptorPool>(quartile_desc_layout);
        filter_desc_pool = std::make_shared<DescriptorPool>(filter_desc_layout);

        quartile_set = std::make_shared<DescriptorSet>(quartile_desc_pool);
        filter_set = std::make_shared<DescriptorSet>(filter_desc_pool);
    }

    quartile_group_count_x =
        (irr_create_info.extent.width + quartile_local_size_x - 1) / quartile_local_size_x;
    quartile_group_count_y =
        (irr_create_info.extent.height + quartile_local_size_y - 1) / quartile_local_size_y;
    filter_group_count_x =
        (irr_create_info.extent.width + filter_local_size_x - 1) / filter_local_size_x;
    filter_group_count_y =
        (irr_create_info.extent.height + filter_local_size_y - 1) / filter_local_size_y;

    vk::ImageCreateInfo quartile_image_create_info = irr_create_info;
    quartile_image_create_info.usage |= vk::ImageUsageFlagBits::eSampled;
    quartile_image_create_info.setExtent({quartile_group_count_x, quartile_group_count_y, 1});
    const ImageHandle quartile_image = allocator->createImage(quartile_image_create_info);
    vk::ImageViewCreateInfo quartile_image_view_create_info{
        {}, *quartile_image,        vk::ImageViewType::e2D, quartile_image->get_format(),
        {}, first_level_and_layer()};
    quartile_texture = allocator->createTexture(quartile_image, quartile_image_view_create_info);
    quartile_texture->attach_sampler(allocator->get_sampler_pool()->linear_mirrored_repeat());

    DescriptorSetUpdate(quartile_set)
        .write_descriptor_texture(0, quartile_texture, 0, 1, vk::ImageLayout::eGeneral)
        .update(context);
    DescriptorSetUpdate(filter_set)
        .write_descriptor_texture(0, quartile_texture, 0, 1,
                                  vk::ImageLayout::eShaderReadOnlyOptimal)
        .update(context);

    auto quartile_pipe_layout = PipelineLayoutBuilder(context)
                                    .add_descriptor_set_layout(graph_layout)
                                    .add_descriptor_set_layout(quartile_desc_layout)
                                    .add_push_constant<QuartilePushConstant>()
                                    .build_pipeline_layout();

    auto quartile_spec_builder = SpecializationInfoBuilder();
    quartile_spec_builder.add_entry(quartile_local_size_x, quartile_local_size_y);
    auto quartile_spec = quartile_spec_builder.build();
    quartile =
        std::make_shared<ComputePipeline>(quartile_pipe_layout, quartile_module, quartile_spec);

    auto filter_pipe_layout = PipelineLayoutBuilder(context)
                                  .add_descriptor_set_layout(graph_layout)
                                  .add_descriptor_set_layout(filter_desc_layout)
                                  .add_push_constant<FilterPushConstant>()
                                  .build_pipeline_layout();
    auto filter_spec_builder = SpecializationInfoBuilder();
    const uint32_t wg_rounded_irr_size_x = quartile_group_count_x * quartile_local_size_x;
    const uint32_t wg_rounded_irr_size_y = quartile_group_count_y * quartile_local_size_y;
    filter_spec_builder.add_entry(filter_local_size_x, filter_local_size_y, wg_rounded_irr_size_x,
                                  wg_rounded_irr_size_y);
    auto filter_spec = filter_spec_builder.build();
    filter = std::make_shared<ComputePipeline>(filter_pipe_layout, filter_module, filter_spec);
}

void FireflyFilterNode::cmd_process(
    const vk::CommandBuffer& cmd,
    [[maybe_unused]] GraphRun& run,
    const uint32_t set_index,
    [[maybe_unused]] const std::vector<ImageHandle>& image_inputs,
    [[maybe_unused]] const std::vector<BufferHandle>& buffer_inputs,
    [[maybe_unused]] const std::vector<ImageHandle>& image_outputs,
    [[maybe_unused]] const std::vector<BufferHandle>& buffer_outputs) {

    if (filter_pc.enabled) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "compute quartiles");
        auto bar = quartile_texture->get_image()->barrier(
            vk::ImageLayout::eGeneral, {}, vk::AccessFlagBits::eShaderWrite,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), true);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, bar);

        quartile->bind(cmd);
        quartile->bind_descriptor_set(cmd, graph_sets[set_index], 0);
        quartile->bind_descriptor_set(cmd, quartile_set, 1);
        quartile->push_constant(cmd, quartile_pc);
        cmd.dispatch(quartile_group_count_x, quartile_group_count_y, 1);
    }

    auto bar = quartile_texture->get_image()->barrier(
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers());
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, bar);

    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "filter");
        filter->bind(cmd);
        filter->bind_descriptor_set(cmd, graph_sets[set_index], 0);
        filter->bind_descriptor_set(cmd, filter_set, 1);
        filter->push_constant(cmd, filter_pc);
        cmd.dispatch(filter_group_count_x, filter_group_count_y, 1);
    }
}

void FireflyFilterNode::get_configuration(Configuration& config, bool&) {
    config.config_bool("enable", filter_pc.enabled);

    config.config_float("bias", filter_pc.bias, "Adds this value to the maximum allowed luminance.",
                        0.1);
    config.config_float("IPR factor", filter_pc.ipr_factor,
                        "Inter-percentile range factor. Increase to allow higher outliers.");

    config.st_separate();
    config.config_percent("percentile lower", quartile_pc.percentile_lower);
    config.config_percent("percentile upper", quartile_pc.percentile_upper);
}

} // namespace merian
