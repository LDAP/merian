#include "accumulate.hpp"

#include "accumulate.comp.spv.h"
#include "calculate_percentiles.comp.spv.h"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

AccumulateNode::AccumulateNode(const SharedContext context,
                               const ResourceAllocatorHandle allocator,
                               const std::optional<vk::Format> format)
    : context(context), allocator(allocator), format(format) {
    percentile_module =
        std::make_shared<ShaderModule>(context, merian_calculate_percentiles_comp_spv_size(),
                                       merian_calculate_percentiles_comp_spv());
    accumulate_module = std::make_shared<ShaderModule>(context, merian_accumulate_comp_spv_size(),
                                                       merian_accumulate_comp_spv());
}

AccumulateNode::~AccumulateNode() {}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
AccumulateNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("prev_accum", 1),
            NodeInputDescriptorImage::compute_read("prev_moments", 1),

            NodeInputDescriptorImage::compute_read("irr"),

            NodeInputDescriptorImage::compute_read("mv"),
            NodeInputDescriptorImage::compute_read("moments_in"),
        },
        {
            NodeInputDescriptorBuffer::compute_read("gbuf"),
            NodeInputDescriptorBuffer::compute_read("prev_gbuf", 1),
        },
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
AccumulateNode::describe_outputs(
    const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
    const std::vector<NodeOutputDescriptorBuffer>&) {

    irr_create_info = connected_image_outputs[2].create_info;
    const auto moments_create_info = connected_image_outputs[4].create_info;

    return {
        {
            NodeOutputDescriptorImage::compute_write(
                "out_irr", format.value_or(irr_create_info.format), irr_create_info.extent),
            NodeOutputDescriptorImage::compute_write("out_moments", moments_create_info.format,
                                                     moments_create_info.extent),
        },
        {},
    };
}

void AccumulateNode::cmd_build([[maybe_unused]] const vk::CommandBuffer& cmd,
                               const std::vector<std::vector<ImageHandle>>& image_inputs,
                               const std::vector<std::vector<BufferHandle>>& buffer_inputs,
                               const std::vector<std::vector<ImageHandle>>& image_outputs,
                               const std::vector<std::vector<BufferHandle>>& buffer_outputs) {
    std::tie(graph_textures, graph_sets, graph_pool, graph_layout) =
        make_graph_descriptor_sets(context, allocator, image_inputs, buffer_inputs, image_outputs,
                                   buffer_outputs, graph_layout);

    if (!percentile_desc_layout) {
        percentile_desc_layout =
            DescriptorSetLayoutBuilder().add_binding_storage_image().build_layout(context);
        accumulate_desc_layout =
            DescriptorSetLayoutBuilder().add_binding_combined_sampler().build_layout(context);

        percentile_desc_pool = std::make_shared<DescriptorPool>(percentile_desc_layout);
        accumulate_desc_pool = std::make_shared<DescriptorPool>(accumulate_desc_layout);

        percentile_set = std::make_shared<DescriptorSet>(percentile_desc_pool);
        accumulate_set = std::make_shared<DescriptorSet>(accumulate_desc_pool);
    }

    percentile_group_count_x =
        (irr_create_info.extent.width + percentile_local_size_x - 1) / percentile_local_size_x;
    percentile_group_count_y =
        (irr_create_info.extent.height + percentile_local_size_y - 1) / percentile_local_size_y;
    filter_group_count_x =
        (irr_create_info.extent.width + filter_local_size_x - 1) / filter_local_size_x;
    filter_group_count_y =
        (irr_create_info.extent.height + filter_local_size_y - 1) / filter_local_size_y;

    vk::ImageCreateInfo quartile_image_create_info = irr_create_info;
    quartile_image_create_info.usage |= vk::ImageUsageFlagBits::eSampled;
    quartile_image_create_info.setExtent({percentile_group_count_x, percentile_group_count_y, 1});
    const ImageHandle quartile_image = allocator->createImage(quartile_image_create_info);
    vk::ImageViewCreateInfo quartile_image_view_create_info{
        {}, *quartile_image,        vk::ImageViewType::e2D, quartile_image->get_format(),
        {}, first_level_and_layer()};
    percentile_texture = allocator->createTexture(quartile_image, quartile_image_view_create_info);
    percentile_texture->attach_sampler(allocator->get_sampler_pool()->linear_mirrored_repeat());

    DescriptorSetUpdate(percentile_set)
        .write_descriptor_texture(0, percentile_texture, 0, 1, vk::ImageLayout::eGeneral)
        .update(context);
    DescriptorSetUpdate(accumulate_set)
        .write_descriptor_texture(0, percentile_texture, 0, 1,
                                  vk::ImageLayout::eShaderReadOnlyOptimal)
        .update(context);

    auto quartile_pipe_layout = PipelineLayoutBuilder(context)
                                    .add_descriptor_set_layout(graph_layout)
                                    .add_descriptor_set_layout(percentile_desc_layout)
                                    .add_push_constant<QuartilePushConstant>()
                                    .build_pipeline_layout();

    auto quartile_spec_builder = SpecializationInfoBuilder();
    quartile_spec_builder.add_entry(percentile_local_size_x, percentile_local_size_y);
    auto quartile_spec = quartile_spec_builder.build();
    calculate_percentiles =
        std::make_shared<ComputePipeline>(quartile_pipe_layout, percentile_module, quartile_spec);

    auto filter_pipe_layout = PipelineLayoutBuilder(context)
                                  .add_descriptor_set_layout(graph_layout)
                                  .add_descriptor_set_layout(accumulate_desc_layout)
                                  .add_push_constant<FilterPushConstant>()
                                  .build_pipeline_layout();
    auto filter_spec_builder = SpecializationInfoBuilder();
    const uint32_t wg_rounded_irr_size_x = percentile_group_count_x * percentile_local_size_x;
    const uint32_t wg_rounded_irr_size_y = percentile_group_count_y * percentile_local_size_y;
    filter_spec_builder.add_entry(filter_local_size_x, filter_local_size_y, wg_rounded_irr_size_x,
                                  wg_rounded_irr_size_y, filter_mode, extended_search,
                                  reuse_border);
    auto filter_spec = filter_spec_builder.build();
    accumulate =
        std::make_shared<ComputePipeline>(filter_pipe_layout, accumulate_module, filter_spec);
}

void AccumulateNode::cmd_process(const vk::CommandBuffer& cmd,
                                 [[maybe_unused]] GraphRun& run,
                                 const uint32_t set_index,
                                 [[maybe_unused]] const std::vector<ImageHandle>& image_inputs,
                                 [[maybe_unused]] const std::vector<BufferHandle>& buffer_inputs,
                                 [[maybe_unused]] const std::vector<ImageHandle>& image_outputs,
                                 [[maybe_unused]] const std::vector<BufferHandle>& buffer_outputs) {

    if (accumulate_pc.firefly_filter_enable || accumulate_pc.adaptive_alpha_reduction > 0.0f) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "compute percentiles");
        auto bar = percentile_texture->get_image()->barrier(
            vk::ImageLayout::eGeneral, {}, vk::AccessFlagBits::eShaderWrite,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), true);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, bar);

        calculate_percentiles->bind(cmd);
        calculate_percentiles->bind_descriptor_set(cmd, graph_sets[set_index], 0);
        calculate_percentiles->bind_descriptor_set(cmd, percentile_set, 1);
        calculate_percentiles->push_constant(cmd, percentile_pc);
        cmd.dispatch(percentile_group_count_x, percentile_group_count_y, 1);
    }

    auto bar = percentile_texture->get_image()->barrier(
        vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        all_levels_and_layers());
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, bar);

    {
        accumulate_pc.clear = run.get_iteration() == 0 || clear;
        clear = false;

        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "accumulate");
        accumulate->bind(cmd);
        accumulate->bind_descriptor_set(cmd, graph_sets[set_index], 0);
        accumulate->bind_descriptor_set(cmd, accumulate_set, 1);
        accumulate->push_constant(cmd, accumulate_pc);
        cmd.dispatch(filter_group_count_x, filter_group_count_y, 1);
    }
}

void AccumulateNode::get_configuration(Configuration& config, bool& needs_rebuild) {
    config.st_separate("Accumulation");
    config.config_float("alpha", accumulate_pc.accum_alpha, 0, 1,
                        "Blend factor with the previous information. More means more reuse");
    config.config_float("max history", accumulate_pc.accum_max_hist,
                        "artificially limit the history counter. This can be a good alternative to "
                        "reducing the blend alpha");
    config.st_no_space();
    accumulate_pc.accum_max_hist =
        config.config_bool("inf history") ? INFINITY : accumulate_pc.accum_max_hist;
    clear = config.config_bool("clear");

    config.st_separate("Reproject");
    float angle = glm::acos(accumulate_pc.normal_reject_cos);
    config.config_angle("normal threshold", angle, "Reject points with normals farther apart", 0,
                        180);
    accumulate_pc.normal_reject_cos = glm::cos(angle);
    config.config_percent("depth threshold", accumulate_pc.depth_reject_percent,
                          "Reject points with depths farther apart (relative to the max)");
    int old_filter_mode = filter_mode;
    config.config_options("filter mode", filter_mode, {"nearest", "linear"});
    needs_rebuild |= old_filter_mode != filter_mode;
    const int old_extended_search = extended_search;
    const int old_reuse_border = reuse_border;
    config.config_bool("extended search", extended_search,
                       "search in a 3x3 radius with weakened rejection thresholds for valid "
                       "information if nothing was found. Helps "
                       "with artifacts at edges");
    config.config_bool("reuse border", reuse_border,
                       "Reuse border information (if valid) for pixel where the motion vector "
                       "points outside of the image. Can lead to smearing.");
    needs_rebuild |= old_extended_search != extended_search || old_reuse_border != reuse_border;

    config.st_separate("Firefly Suppression");
    config.config_bool("firefly filter enable", accumulate_pc.firefly_filter_enable);

    config.config_float("firefly filter bias", accumulate_pc.firefly_bias,
                        "Adds this value to the maximum allowed luminance.", 0.1);
    config.config_float("IPR factor", accumulate_pc.firefly_ipr_factor,
                        "Inter-percentile range factor. Increase to allow higher outliers.");
    config.st_separate();
    config.config_percent("firefly percentile lower", percentile_pc.firefly_percentile_lower);
    config.config_percent("firefly percentile upper", percentile_pc.firefly_percentile_upper);
    config.st_separate();
    config.config_float("hard clamp", accumulate_pc.firefly_hard_clamp, "DANGER: Introduces bias",
                        0.1);
    config.st_no_space();
    if (config.config_bool("inf clamp"))
        accumulate_pc.firefly_hard_clamp = INFINITY;

    config.st_separate("Adaptive alpha reduction");
    config.config_percent("adaptivity", accumulate_pc.adaptive_alpha_reduction,
                          "(1. - adaptivity) is the smallest factor that alpha is multipied with");
    config.config_float(
        "adaptivity IPR factor", accumulate_pc.adaptive_alpha_ipr_factor,
        "Inter-percentile range for adaptive reduction. Increase to soften reduction.", 0.1);
    config.st_separate();
    config.config_percent("adaptivity percentile lower",
                          percentile_pc.adaptive_alpha_percentile_lower);
    config.config_percent("adaptivity percentile upper",
                          percentile_pc.adaptive_alpha_percentile_upper);
}

void AccumulateNode::request_clear() {
    clear = true;
}

} // namespace merian
