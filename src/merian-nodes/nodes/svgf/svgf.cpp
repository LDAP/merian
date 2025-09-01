#include "merian-nodes/nodes/svgf/svgf.hpp"
#include "merian-nodes/nodes/svgf/config.h"
#include "merian/utils/math.hpp"
#include "merian/vk/descriptors/descriptor_set_layout_builder.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/shader/slang_session.hpp"

namespace merian_nodes {

SVGF::SVGF(const ContextHandle& context,
           const ResourceAllocatorHandle& allocator,
           const std::optional<vk::Format> output_format)
    : Node(), context(context), allocator(allocator), output_format(output_format) {}

SVGF::~SVGF() {}

std::vector<InputConnectorHandle> SVGF::describe_inputs() {
    return {
        con_prev_out, con_src, con_history, con_albedo, con_mv, con_gbuffer, con_prev_gbuffer,
    };
}

std::vector<OutputConnectorHandle> SVGF::describe_outputs(const NodeIOLayout& io_layout) {
    irr_create_info = io_layout[con_src]->get_create_info_or_throw();
    if (output_format)
        irr_create_info.format = output_format.value();

    con_out =
        ManagedVkImageOut::compute_write("out", irr_create_info.format, irr_create_info.extent);

    return {con_out};
}

SVGF::NodeStatusFlags SVGF::on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                         const DescriptorSetLayoutHandle& graph_layout) {
    variance_estimate_local_size = workgroup_size_for_shared_memory_with_halo(
        context, VE_SHARED_MEMORY_PER_PIXEL, SVGF_VE_HALO_RADIUS, 32, 16);
    if (kaleidoscope && kaleidoscope_use_shmem) {
        filter_local_size = workgroup_size_for_shared_memory_with_halo(
            context, FILTER_SHARED_MEMORY_PER_PIXEL, SVGF_FILTER_HALO_RADIUS, 32, 16);
    } else {
        filter_local_size = 32;
    }

    if (!ping_pong_layout) {
        ping_pong_layout = DescriptorSetLayoutBuilder()
                               .add_binding_combined_sampler() // irradiance
                               .add_binding_storage_image()
                               .add_binding_combined_sampler() // gbuffer
                               .add_binding_storage_image()
                               .build_layout(context);
    }
    filter_pool = std::make_shared<DescriptorPool>(ping_pong_layout, 2); // ping pong

    // Ping pong textures
    irr_create_info.usage |= vk::ImageUsageFlagBits::eSampled;
    if (kaleidoscope) {
        const uint32_t padding_multiple =
            std::max((uint32_t)1 << (std::max(svgf_iterations, 1) - 1), filter_local_size);
        irr_create_info.extent.width = round_up(irr_create_info.extent.width, padding_multiple);
        irr_create_info.extent.height = round_up(irr_create_info.extent.height, padding_multiple);
        SPDLOG_DEBUG("SVGF padding to {} -> ({} x {})", padding_multiple,
                     irr_create_info.extent.width, irr_create_info.extent.height);
    }

    for (int i = 0; i < 2; i++) {
        if (!ping_pong_res[i].set)
            ping_pong_res[i].set = std::make_shared<DescriptorSet>(filter_pool);

        // irradiance
        ImageHandle tmp_irr_image = allocator->createImage(irr_create_info, MemoryMappingType::NONE,
                                                           fmt::format("SVGF ping pong: {}", i));
        ping_pong_res[i].ping_pong =
            allocator->createTexture(tmp_irr_image, tmp_irr_image->make_view_create_info(),
                                     allocator->get_sampler_pool()->linear_mirrored_repeat());

        // gbuffer
        vk::ImageCreateInfo gbuf_create_info = irr_create_info;
        if (!kaleidoscope) {
            // keep to prevent special cases but reduce to single pixel...
            gbuf_create_info.extent.width = 1;
            gbuf_create_info.extent.height = 1;
        }
        gbuf_create_info.format = vk::Format::eR32G32B32A32Uint;
        ImageHandle tmp_gbuf_image = allocator->createImage(gbuf_create_info);
        ping_pong_res[i].gbuf_ping_pong =
            allocator->createTexture(tmp_gbuf_image, tmp_gbuf_image->make_view_create_info(),
                                     allocator->get_sampler_pool()->nearest_mirrored_repeat());
    }
    for (int i = 0; i < 2; i++) {
        ping_pong_res[i]
            .set
            ->queue_descriptor_write_texture(0, ping_pong_res[i].ping_pong, 0,
                                             vk::ImageLayout::eShaderReadOnlyOptimal)
            .queue_descriptor_write_texture(1, ping_pong_res[i ^ 1].ping_pong, 0,
                                            vk::ImageLayout::eGeneral)
            .queue_descriptor_write_texture(2, ping_pong_res[i].gbuf_ping_pong, 0,
                                            vk::ImageLayout::eShaderReadOnlyOptimal)
            .queue_descriptor_write_texture(3, ping_pong_res[i ^ 1].gbuf_ping_pong, 0,
                                            vk::ImageLayout::eGeneral)
            .update();
    }

    {
        const GLSLShaderCompilerHandle compiler = GLSLShaderCompiler::get();
        CompilationSessionDescription compilation_session_desc(context);
        compilation_session_desc.set_preprocessor_macro("FILTER_TYPE", std::to_string(filter_type));
        if (kaleidoscope) {
            compilation_session_desc.set_preprocessor_macro("KALEIDOSCOPE", "1");
            if (kaleidoscope_use_shmem) {
                compilation_session_desc.set_preprocessor_macro("KALEIDOSCOPE_USE_SHMEM", "1");
            }
        }
        compilation_session_desc.add_search_path("merian-nodes/nodes/svgf");

        merian::SlangSession slang_session(compilation_session_desc);

        filter_module = slang_session.load_module_from_path_and_compile_entry_point(
            context, "merian-nodes/nodes/svgf/svgf_filter.slang");
        taa_module = slang_session.load_module_from_path_and_compile_entry_point(
            context, "merian-nodes/nodes/svgf/svgf_taa.slang");
        variance_estimate_module = slang_session.load_module_from_path_and_compile_entry_point(
            context, "merian-nodes/nodes/svgf/svgf_variance_estimate.slang");

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
            spec_builder.add_entry(variance_estimate_local_size, variance_estimate_local_size,
                                   svgf_iterations);
            SpecializationInfoHandle variance_estimate_spec = spec_builder.build();
            variance_estimate = ComputePipeline::create(
                variance_estimate_pipe_layout, variance_estimate_module, variance_estimate_spec);
        }
        {
            filters.clear();
            filters.resize(svgf_iterations);
            for (int i = 0; i < svgf_iterations; i++) {
                auto spec_builder = SpecializationInfoBuilder();
                int gap = 1 << i;
                spec_builder.add_entry(filter_local_size, filter_local_size, gap, i,
                                       svgf_iterations - 1);
                SpecializationInfoHandle filter_spec = spec_builder.build();
                filters[i] = ComputePipeline::create(filter_pipe_layout, filter_module,
                                                               filter_spec);
            }
        }
        {
            auto spec_builder = SpecializationInfoBuilder();
            spec_builder.add_entry(taa_local_size, taa_local_size, taa_debug, taa_filter_prev,
                                   taa_clamping, taa_mv_sampling,
                                   enable_mv && io_layout.is_connected(con_mv));
            SpecializationInfoHandle taa_spec = spec_builder.build();
            taa = ComputePipeline::create(taa_pipe_layout, taa_module, taa_spec);
        }
    }

    return {};
}

void SVGF::process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();

    // PREPARE (VARIANCE ESTIMATE)
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "estimate variance");
        // prepare image to write to
        auto bar = ping_pong_res[0].ping_pong->get_image()->barrier(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead,
            vk::AccessFlagBits::eShaderWrite, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers(), true);
        auto gbuf_bar = ping_pong_res[0].gbuf_ping_pong->get_image()->barrier(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead,
            vk::AccessFlagBits::eShaderWrite, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers(), true);
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader, {bar, gbuf_bar});

        // run kernel
        cmd->bind(variance_estimate);
        cmd->bind_descriptor_set(variance_estimate, descriptor_set, 0);
        cmd->bind_descriptor_set(variance_estimate, ping_pong_res[1].set, 1);
        VarianceEstimatePushConstant precomputed_variance_estimate_pc = variance_estimate_pc;
        precomputed_variance_estimate_pc.depth_accept =
            -10.0f / precomputed_variance_estimate_pc.depth_accept;
        cmd->push_constant(variance_estimate, precomputed_variance_estimate_pc);
        cmd->dispatch(irr_create_info.extent, variance_estimate_local_size,
                      variance_estimate_local_size);

        // make sure writes are visible
        bar = ping_pong_res[0].ping_pong->get_image()->barrier(
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers());
        gbuf_bar = ping_pong_res[0].gbuf_ping_pong->get_image()->barrier(
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers());
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader, {bar, gbuf_bar});
    }

    // FILTER
    DescriptorSetHandle read_set = ping_pong_res[0].set;
    for (int i = 0; i < svgf_iterations; i++) {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, fmt::format("filter iteration {}", i));
        EAWRes& write_res = ping_pong_res[!(i & 1)];

        // prepare image to write to
        auto bar = write_res.ping_pong->get_image()->barrier(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead,
            vk::AccessFlagBits::eShaderWrite, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers(), true);
        auto gbuf_bar = write_res.gbuf_ping_pong->get_image()->barrier(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead,
            vk::AccessFlagBits::eShaderWrite, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers(), true);
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader, {bar, gbuf_bar});

        // run filter
        cmd->bind(filters[i]);
        cmd->bind_descriptor_set(filters[i], descriptor_set, 0);
        cmd->bind_descriptor_set(filters[i], read_set, 1);
        FilterPushConstant precomputed_filter_pc = filter_pc;
        precomputed_filter_pc.param_z = -10.0f / precomputed_filter_pc.param_z;
        cmd->push_constant(filters[i], precomputed_filter_pc);
        cmd->dispatch(irr_create_info.extent, filter_local_size, filter_local_size);

        bar = write_res.ping_pong->get_image()->barrier(
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers());
        gbuf_bar = write_res.gbuf_ping_pong->get_image()->barrier(
            vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers());
        cmd->barrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eComputeShader, {bar, gbuf_bar});

        read_set = write_res.set;
    }

    // TAA
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "taa");
        cmd->bind(taa);
        cmd->bind_descriptor_set(taa, descriptor_set, 0);
        cmd->bind_descriptor_set(taa, read_set, 1);
        cmd->push_constant(taa, taa_pc);
        cmd->dispatch(io[con_out]->get_extent(), taa_local_size, taa_local_size);
    }
}

SVGF::NodeStatusFlags SVGF::properties(Properties& config) {
    bool needs_rebuild = false;

    config.st_separate("Variance estimate");
    config.config_float("spatial falloff", variance_estimate_pc.spatial_falloff,
                        "higher means only use spatial with very low history", 0.01);
    config.config_float("spatial bias", variance_estimate_pc.spatial_bias,
                        "higher means use spatial information longer before using the falloff",
                        0.1);
    float angle = glm::acos(variance_estimate_pc.normal_reject_cos);
    config.config_angle("normal reject", angle, "Reject points with farther apart", 0, 180);
    variance_estimate_pc.normal_reject_cos = glm::cos(angle);
    config.config_float("depth accept", variance_estimate_pc.depth_accept, "More means more reuse");

    config.st_separate("Filter");
    needs_rebuild |= config.config_int("SVGF iterations", svgf_iterations, 0, 10,
                                       "0 disables SVGF completely (TAA-only mode)");
    config.config_float("filter depth", filter_pc.param_z, "more means more blur");
    angle = glm::acos(filter_pc.param_n);
    config.config_angle("filter normals", angle, "Reject with normals farther apart", 0, 180);
    filter_pc.param_n = glm::cos(angle);
    config.config_float("filter luminance", filter_pc.param_l, "more means more blur", 0.1);
    config.config_float("z-bias normals", filter_pc.z_bias_normals,
                        "z-dependent rejection: increase to reject more. Disable with <= 0.");
    config.config_float("z-bias depth", filter_pc.z_bias_depth,
                        "z-dependent rejection: increase to reject more. Disable with <= 0.");
    if (config.config_options("filter type", filter_type, {"atrous", "box", "subsampled"},
                              Properties::OptionsStyle::COMBO)) {
        needs_rebuild = true;
        kaleidoscope = filter_type == 0;
    }

    needs_rebuild |= config.config_bool("kaleidoscope", kaleidoscope);
    needs_rebuild |= config.config_bool("kaleidoscope: shmem", kaleidoscope_use_shmem,
                                        "use shared memory for kaleidoscope");

    config.st_separate("TAA");
    config.config_float(
        "TAA alpha", taa_pc.blend_alpha, 0, 1,
        "Blend factor for the final image and the previous image. More means more reuse.");

    needs_rebuild |=
        config.config_bool("enable motion vectors", enable_mv, "uses motion vectors if connected.");
    if (enable_mv) {
        needs_rebuild |=
            config.config_options("mv sampling", taa_mv_sampling, {"center", "magnitude dilation"},
                                  Properties::OptionsStyle::COMBO);
    }
    needs_rebuild |= config.config_options("filter", taa_filter_prev, {"none", "catmull rom"},
                                           Properties::OptionsStyle::COMBO);
    needs_rebuild |= config.config_options("clamping", taa_clamping, {"min-max", "moments"},
                                           Properties::OptionsStyle::COMBO);
    if (taa_clamping == 1)
        config.config_float(
            "TAA rejection threshold", taa_pc.rejection_threshold,
            "TAA rejection threshold for the previous frame, in units of standard deviation", 0.01);
    needs_rebuild |= config.config_options("debug", taa_debug,
                                           {"none", "irradiance", "variance", "normal", "depth",
                                            "albedo", "grad z", "irradiance nan/inf", "mv"});

    config.st_separate();
    config.output_text("local size variance estimate: {}\nlocal size filter: {}",
                       variance_estimate_local_size, filter_local_size);

    if (needs_rebuild) {
        return NEEDS_RECONNECT;
    }

    return {};
}

} // namespace merian_nodes
