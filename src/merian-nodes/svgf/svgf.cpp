#include "svgf.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "svgf_filter.comp.spv.h"
#include "svgf_taa.comp.spv.h"
#include "svgf_variance_estimate.comp.spv.h"

namespace merian {

SVGFNode::SVGFNode(const SharedContext context,
                   const ResourceAllocatorHandle allocator,
                   const std::optional<vk::Format> output_format)
    : context(context), allocator(allocator), output_format(output_format) {
    variance_estimate_module =
        std::make_shared<ShaderModule>(context, merian_svgf_variance_estimate_comp_spv_size(),
                                       merian_svgf_variance_estimate_comp_spv());
    filter_module = std::make_shared<ShaderModule>(context, merian_svgf_filter_comp_spv_size(),
                                                   merian_svgf_filter_comp_spv());
    taa_module = std::make_shared<ShaderModule>(context, merian_svgf_taa_comp_spv_size(),
                                                merian_svgf_taa_comp_spv());
}

SVGFNode::~SVGFNode() {}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
SVGFNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("prev_out", 1),

            NodeInputDescriptorImage::compute_read("irr"),
            NodeInputDescriptorImage::compute_read("moments"),

            NodeInputDescriptorImage::compute_read("albedo"),
            NodeInputDescriptorImage::compute_read("mv"),
        },
        {
            NodeInputDescriptorBuffer::compute_read("gbuffer"),
            NodeInputDescriptorBuffer::compute_read("prev_gbuffer", 1),
        },
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
SVGFNode::describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                           const std::vector<NodeOutputDescriptorBuffer>&) {
    // clang-format off
    irr_create_info = connected_image_outputs[1].create_info;
    if (output_format)
        irr_create_info.format = output_format.value();

    return {
        {
            NodeOutputDescriptorImage::compute_write("out", irr_create_info.format, irr_create_info.extent),
        },
        {},
    };
    // clang-format on
}

void SVGFNode::cmd_build([[maybe_unused]] const vk::CommandBuffer& cmd,
                         const std::vector<std::vector<ImageHandle>>& image_inputs,
                         const std::vector<std::vector<BufferHandle>>& buffer_inputs,
                         const std::vector<std::vector<ImageHandle>>& image_outputs,
                         const std::vector<std::vector<BufferHandle>>& buffer_outputs) {
    std::tie(graph_textures, graph_sets, graph_pool, graph_layout) =
        make_graph_descriptor_sets(context, allocator, image_inputs, buffer_inputs, image_outputs,
                                   buffer_outputs, graph_layout);
    if (!ping_pong_layout) {
        ping_pong_layout = DescriptorSetLayoutBuilder()
                               .add_binding_combined_sampler()
                               .add_binding_storage_image()
                               .build_layout(context);
    }
    filter_pool = std::make_shared<DescriptorPool>(ping_pong_layout, 2); // ping pong

    // Ping pong textures
    irr_create_info.usage |= vk::ImageUsageFlagBits::eSampled;
    for (int i = 0; i < 2; i++) {
        if (!ping_pong_res[i].set)
            ping_pong_res[i].set = std::make_shared<DescriptorSet>(filter_pool);

        ImageHandle tmp_irr_image = allocator->createImage(irr_create_info);
        vk::ImageViewCreateInfo create_image_view{
            {}, *tmp_irr_image,         vk::ImageViewType::e2D, tmp_irr_image->get_format(),
            {}, first_level_and_layer()};
        ping_pong_res[i].ping_pong = allocator->createTexture(tmp_irr_image, create_image_view);
        ping_pong_res[i].ping_pong->attach_sampler(
            allocator->get_sampler_pool()->linear_mirrored_repeat());
    }
    for (int i = 0; i < 2; i++) {
        DescriptorSetUpdate(ping_pong_res[i].set)
            .write_descriptor_texture(0, ping_pong_res[i].ping_pong, 0, 1,
                                      vk::ImageLayout::eShaderReadOnlyOptimal)
            .write_descriptor_texture(1, ping_pong_res[i ^ 1].ping_pong, 0, 1,
                                      vk::ImageLayout::eGeneral)
            .update(context);
    }

    {
        auto variance_estimate_pipe_layout = PipelineLayoutBuilder(context)
                                                 .add_descriptor_set_layout(graph_layout)
                                                 .add_descriptor_set_layout(ping_pong_layout)
                                                 .add_push_constant<VarianceEstimatePushConstant>()
                                                 .build_pipeline_layout();
        auto filter_pipe_layout = PipelineLayoutBuilder(context)
                                      .add_descriptor_set_layout(graph_layout)
                                      .add_descriptor_set_layout(ping_pong_layout)
                                      .add_push_constant<FilterPushConstant>()
                                      .build_pipeline_layout();
        auto taa_pipe_layout = PipelineLayoutBuilder(context)
                                   .add_descriptor_set_layout(graph_layout)
                                   .add_descriptor_set_layout(ping_pong_layout)
                                   .add_push_constant<TAAPushConstant>()
                                   .build_pipeline_layout();

        {
            auto spec_builder = SpecializationInfoBuilder();
            spec_builder.add_entry(local_size_x, local_size_y);
            SpecializationInfoHandle variance_estimate_spec = spec_builder.build();
            variance_estimate = std::make_shared<ComputePipeline>(
                variance_estimate_pipe_layout, variance_estimate_module, variance_estimate_spec);
        }
        {
            filters.clear();
            filters.resize(svgf_iterations);
            for (int i = 0; i < svgf_iterations; i++) {
                auto spec_builder = SpecializationInfoBuilder();
                int gap = 1 << i;
                spec_builder.add_entry(local_size_x, local_size_y, gap, filter_variance,
                                       filter_type, i);
                SpecializationInfoHandle taa_spec = spec_builder.build();
                filters[i] =
                    std::make_shared<ComputePipeline>(filter_pipe_layout, filter_module, taa_spec);
            }
        }
        {
            auto spec_builder = SpecializationInfoBuilder();
            spec_builder.add_entry(local_size_x, local_size_y, taa_debug, taa_filter_prev,
                                   taa_clamping, taa_mv_sampling);
            SpecializationInfoHandle taa_spec = spec_builder.build();
            taa = std::make_shared<ComputePipeline>(taa_pipe_layout, taa_module, taa_spec);
        }
    }

    group_count_x = (irr_create_info.extent.width + local_size_x - 1) / local_size_x;
    group_count_y = (irr_create_info.extent.height + local_size_y - 1) / local_size_y;
}

void SVGFNode::cmd_process(const vk::CommandBuffer& cmd,
                           [[maybe_unused]] GraphRun& run,
                           const uint32_t set_index,
                           [[maybe_unused]] const std::vector<ImageHandle>& image_inputs,
                           [[maybe_unused]] const std::vector<BufferHandle>& buffer_inputs,
                           [[maybe_unused]] const std::vector<ImageHandle>& image_outputs,
                           [[maybe_unused]] const std::vector<BufferHandle>& buffer_outputs) {
    // PREPARE (VARIANCE ESTIMATE)
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "estimate variance");
        // prepare image to write to
        auto bar = ping_pong_res[0].ping_pong->get_image()->barrier(
            vk::ImageLayout::eGeneral, {}, vk::AccessFlagBits::eShaderWrite,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), true);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, bar);

        // run kernel
        variance_estimate->bind(cmd);
        variance_estimate->bind_descriptor_set(cmd, graph_sets[set_index], 0);
        variance_estimate->bind_descriptor_set(cmd, ping_pong_res[1].set, 1);
        variance_estimate->push_constant(cmd, variance_estimate_pc);
        cmd.dispatch(group_count_x, group_count_y, 1);

        // make sure writes are visible
        bar = ping_pong_res[0].ping_pong->get_image()->barrier(
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers());
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, bar);
    }

    // FILTER
    DescriptorSetHandle read_set = ping_pong_res[0].set;
    for (int i = 0; i < svgf_iterations; i++) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, fmt::format("filter iteration {}", i));
        EAWRes& write_res = ping_pong_res[!(i & 1)];

        // prepare image to write to
        auto bar = write_res.ping_pong->get_image()->barrier(
            vk::ImageLayout::eGeneral, {}, vk::AccessFlagBits::eShaderWrite,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), true);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, bar);

        // run filter
        filters[i]->bind(cmd);
        filters[i]->bind_descriptor_set(cmd, graph_sets[set_index], 0);
        filters[i]->bind_descriptor_set(cmd, read_set, 1);
        filters[i]->push_constant(cmd, filter_pc);
        cmd.dispatch(group_count_x, group_count_y, 1);

        bar = write_res.ping_pong->get_image()->barrier(
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers());
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                            vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, bar);

        read_set = write_res.set;
    }

    // TAA
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "taa");
        taa->bind(cmd);
        taa->bind_descriptor_set(cmd, graph_sets[set_index], 0);
        taa->bind_descriptor_set(cmd, read_set, 1);
        taa->push_constant(cmd, taa_pc);
        cmd.dispatch(group_count_x, group_count_y, 1);
    }
}

void SVGFNode::get_configuration(Configuration& config, bool& needs_rebuild) {
    config.st_separate("Variance estimate");
    config.config_int("spatial threshold", variance_estimate_pc.spatial_threshold, 0, 120,
                      "Compute the variance spatially for shorter histories.");
    config.config_float("spatial boost", variance_estimate_pc.spatial_variance_boost,
                        "Boost the variance of spatial variance estimates.");
    float angle = glm::acos(variance_estimate_pc.normal_reject_cos);
    config.config_angle("normal reject", angle, "Reject points with farther apart", 0, 90);
    variance_estimate_pc.normal_reject_cos = glm::cos(angle);
    config.config_float("depth accept", variance_estimate_pc.depth_accept, "More means more reuse");

    config.st_separate("Filter");
    const int old_svgf_iterations = svgf_iterations;
    config.config_int("SVGF iterations", svgf_iterations, 0, 10,
                      "0 disables SVGF completely (TAA-only mode)");
    needs_rebuild |= old_svgf_iterations != svgf_iterations;
    config.config_float("filter depth", filter_pc.param_z, "more means more blur");
    angle = glm::acos(filter_pc.param_n);
    config.config_angle("filter normals", angle, "Reject with normals farther apart", 0, 180);
    filter_pc.param_n = glm::cos(angle);
    config.config_float("filter luminance", filter_pc.param_l, "more means more blur", 0.1);
    int old_filter_type = filter_type;
    config.config_options("filter type", filter_type, {"atrous", "box", "subsampled"},
                          Configuration::OptionsStyle::COMBO);
    needs_rebuild |= old_filter_type != filter_type;
    int old_filter_variance = filter_variance;
    config.config_bool("filter variance", filter_variance, "Filter variance with a 3x3 gaussian");
    needs_rebuild |= old_filter_variance != filter_variance;

    config.st_separate("TAA");
    config.config_float(
        "TAA alpha", taa_pc.blend_alpha, 0, 1,
        "Blend factor for the final image and the previous image. More means more reuse.");

    const int old_taa_debug = taa_debug;
    const int old_taa_filter_prev = taa_filter_prev;
    const int old_taa_clamping = taa_clamping;
    const int old_taa_mv_sampling = taa_mv_sampling;
    config.config_options("mv sampling", taa_mv_sampling, {"center", "magnitude dilation"},
                          Configuration::OptionsStyle::COMBO);
    config.config_options("filter", taa_filter_prev, {"none", "catmull rom"},
                          Configuration::OptionsStyle::COMBO);
    config.config_options("clamping", taa_clamping, {"min-max", "moments"},
                          Configuration::OptionsStyle::COMBO);
    if (taa_clamping == 1)
        config.config_float(
            "TAA rejection threshold", taa_pc.rejection_threshold,
            "TAA rejection threshold for the previous frame, in units of standard deviation", 0.01);
    config.config_options("debug", taa_debug,
                          {"none", "variance", "normal", "depth", "albedo", "grad z"});

    needs_rebuild |= old_taa_debug != taa_debug;
    needs_rebuild |= old_taa_filter_prev != taa_filter_prev;
    needs_rebuild |= old_taa_clamping != taa_clamping;
    needs_rebuild |= old_taa_mv_sampling != taa_mv_sampling;
}

} // namespace merian
