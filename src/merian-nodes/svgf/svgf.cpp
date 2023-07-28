#include "svgf.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/graph/node_utils.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

static const uint32_t variance_estimate_spv[] = {
#include "svgf_variance_estimate.comp.spv.h"
};

static const uint32_t filter_spv[] = {
#include "svgf_filter.comp.spv.h"
};

static const uint32_t taa_spv[] = {
#include "svgf_taa.comp.spv.h"
};

namespace merian {

SVGFNode::SVGFNode(const SharedContext context, const ResourceAllocatorHandle allocator)
    : context(context), allocator(allocator) {
    variance_estimate_module = std::make_shared<ShaderModule>(
        context, sizeof(variance_estimate_spv), variance_estimate_spv);
    filter_module = std::make_shared<ShaderModule>(context, sizeof(filter_spv), filter_spv);
    taa_module = std::make_shared<ShaderModule>(context, sizeof(taa_spv), taa_spv);
}

SVGFNode::~SVGFNode() {}

std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
SVGFNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::compute_read("prev_out", 1),

            NodeInputDescriptorImage::compute_read("irr"),
            NodeInputDescriptorImage::compute_read("moments"),

            NodeInputDescriptorImage::compute_read("gbuf"),
            NodeInputDescriptorImage::compute_read("prev_gbuf", 1),

            NodeInputDescriptorImage::compute_read("albedo"),
            NodeInputDescriptorImage::compute_read("mv"),
        },
        {},
    };
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
SVGFNode::describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                           const std::vector<NodeOutputDescriptorBuffer>&) {
    // clang-format off
    irr_create_info = connected_image_outputs[1].create_info;
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

    if (!taa) {
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

        auto spec_builder = SpecializationInfoBuilder();
        spec_builder.add_entry(local_size_x, local_size_y);
        SpecializationInfoHandle spec = spec_builder.build();

        variance_estimate = std::make_shared<ComputePipeline>(variance_estimate_pipe_layout,
                                                              variance_estimate_module, spec);
        filter = std::make_shared<ComputePipeline>(filter_pipe_layout, filter_module, spec);
        taa = std::make_shared<ComputePipeline>(taa_pipe_layout, taa_module, spec);
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
        filter->bind(cmd);
        filter->bind_descriptor_set(cmd, graph_sets[set_index], 0);
        filter->bind_descriptor_set(cmd, read_set, 1);
        filter_pc.gap = 1 << i;
        filter->push_constant(cmd, filter_pc);
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

void SVGFNode::get_configuration(Configuration& config) {
    config.st_separate("Variance estimate");
    config.config_int("spatial threshold", variance_estimate_pc.spatial_threshold, 0, 120,
                      "Compute the variance spatially for shorter histories.");
    config.config_float("spatial boost", variance_estimate_pc.spatial_variance_boost,
                        "Boost the variance of spatial variance estimates.");
    config.config_angle("normal threshold", variance_estimate_pc.normal_reject_rad,
                        "Reject points with normals farther apart", 0, 90);
    config.config_percent("depth threshold", variance_estimate_pc.depth_reject_percent,
                          "Reject points with depths farther apart (relative to the max)");

    config.st_separate("Filter");
    config.config_int("SVGF iterations", svgf_iterations, 0, 10,
                      "0 disables SVGF completely (TAA-only mode)");
    config.config_float("depth sens", filter_pc.param_z, "more means more blur");
    config.config_float("normals sens", filter_pc.param_n, "less means more blur");
    config.config_float("brightness sens", filter_pc.param_l, "more means more blur");
    bool filter_variance = filter_pc.filter_variance;
    config.config_bool("filter variance", filter_variance, "Filter variance with a 3x3 gaussian");
    filter_pc.filter_variance = filter_variance;

    config.st_separate("TAA");
    config.config_float(
        "TAA alpha", taa_pc.blend_alpha, 0, 1,
        "Blend factor for the final image and the previous image. More means more reuse.");
    config.config_options("mv sampling", taa_pc.mv_sampling, {"center", "magnitude dilation"});
    config.config_options("filter", taa_pc.filter_prev, {"none", "catmull rom"}); 
    config.config_options("clamping", taa_pc.clamping, {"min-max", "moments"});
    if (taa_pc.clamping == 1)
        config.config_float(
            "TAA rejection threshold", taa_pc.rejection_threshold,
            "TAA rejection threshold for the previous frame, in units of standard deviation", 0.01);
    bool show_variance = taa_pc.show_variance_estimate;
    config.config_bool("show variance estimate", show_variance);
    taa_pc.show_variance_estimate = show_variance;
}

} // namespace merian
