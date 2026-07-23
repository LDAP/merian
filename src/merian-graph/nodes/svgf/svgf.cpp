#include "merian-graph/nodes/svgf/svgf.hpp"
#include "merian-graph/nodes/svgf/config.h"
#include "merian/utils/math.hpp"

#include "merian/shader/shader_object.hpp"
#include "merian/shader/slang_entry_point.hpp"
#include "merian/shader/spriv_reflect.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

namespace merian {

SVGF::SVGF() {}

SVGF::~SVGF() {}

DeviceSupportInfo SVGF::query_device_support(const DeviceSupportQueryInfo& query_info) {
    ShaderCompileContextHandle compilation_ctx = ShaderCompileContext::create(
        query_info.file_loader->get_search_paths(), query_info.physical_device);
    compilation_ctx->add_search_path("merian-graph/nodes/svgf");

    // Compile the three SVGF shaders
    auto filter_program = SlangProgramEntryPoint::create(compilation_ctx, "svgf_filter.slang");
    auto variance_program =
        SlangProgramEntryPoint::create(compilation_ctx, "svgf_variance_estimate.slang");
    auto taa_program = SlangProgramEntryPoint::create(compilation_ctx, "svgf_taa.slang");

    // Get SPIR-V binaries and reflect to determine requirements
    auto filter_binary = filter_program->get_program()->get_binary();
    auto variance_binary = variance_program->get_program()->get_binary();
    auto taa_binary = taa_program->get_program()->get_binary();

    SpirvReflect filter_reflect(static_cast<const uint32_t*>(filter_binary->getBufferPointer()),
                                filter_binary->getBufferSize());
    SpirvReflect variance_reflect(static_cast<const uint32_t*>(variance_binary->getBufferPointer()),
                                  variance_binary->getBufferSize());
    SpirvReflect taa_reflect(static_cast<const uint32_t*>(taa_binary->getBufferPointer()),
                             taa_binary->getBufferSize());

    // Combine support requirements from all three shaders
    return filter_reflect.query_device_support(query_info) &
           variance_reflect.query_device_support(query_info) &
           taa_reflect.query_device_support(query_info);
}

void SVGF::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
}

std::vector<InputConnectorDescriptor> SVGF::describe_inputs() {
    return {
        {"prev_out", con_prev_out, ConnectorAccess::compute_read, 1},
        {"src", con_src, ConnectorAccess::compute_read},
        {"history", con_history, ConnectorAccess::compute_read},
        {"gbuffer", con_gbuffer, ConnectorAccess::compute_read},
    };
}

std::vector<OutputConnectorDescriptor> SVGF::describe_outputs(const NodeIOLayout& io_layout) {
    irr_create_info = io_layout[con_src]->get_create_info_or_throw();
    if (output_format)
        irr_create_info.format = output_format.value();

    con_out = ManagedVkImageOut::create(irr_create_info.format, irr_create_info.extent);

    return {{"out", con_out, ConnectorAccess::compute_write}};
}

SVGF::NodeStatusFlags SVGF::on_connected([[maybe_unused]] const NodeConnectedInfo& info) {
    const uint32_t max_wg_size_vendor = context->get_physical_device()->is_amd() ? 32 : 16;

    variance_estimate_local_size = workgroup_size_for_shared_memory_with_halo(
        context, VE_SHARED_MEMORY_PER_PIXEL, SVGF_VE_HALO_RADIUS, 32, 16);
    if (kaleidoscope && kaleidoscope_use_shmem) {
        filter_local_size = workgroup_size_for_shared_memory_with_halo(
            context, FILTER_SHARED_MEMORY_PER_PIXEL, SVGF_FILTER_HALO_RADIUS, max_wg_size_vendor,
            16);
    } else {
        filter_local_size = max_wg_size_vendor;
    }
    taa_local_size = max_wg_size_vendor;

    // Ping pong textures
    // internal ping-pong: written as storage, sampled the next iteration (the graph no longer
    // bakes usage into create infos)
    irr_create_info.usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
    if (kaleidoscope) {
        const uint32_t padding_multiple =
            std::max((uint32_t)1 << (std::max(svgf_iterations, 1) - 1), filter_local_size);
        irr_create_info.extent.width = round_up(irr_create_info.extent.width, padding_multiple);
        irr_create_info.extent.height = round_up(irr_create_info.extent.height, padding_multiple);
        SPDLOG_DEBUG("SVGF padding to {} -> ({} x {})", padding_multiple,
                     irr_create_info.extent.width, irr_create_info.extent.height);
    }

    for (int i = 0; i < 2; i++) {
        // irradiance
        ImageHandle tmp_irr_image = allocator->create_image(
            irr_create_info, MemoryMappingType::NONE, fmt::format("SVGF ping pong: {}", i));
        ping_pong_res[i].ping_pong =
            allocator->create_texture(tmp_irr_image, tmp_irr_image->make_view_create_info(),
                                      allocator->get_sampler_pool()->linear_mirrored_repeat());

        // gbuffer
        vk::ImageCreateInfo gbuf_create_info = irr_create_info;
        if (!kaleidoscope) {
            // keep to prevent special cases but reduce to single pixel...
            gbuf_create_info.extent.width = 1;
            gbuf_create_info.extent.height = 1;
        }
        gbuf_create_info.format = vk::Format::eR32G32B32A32Uint;
        ImageHandle tmp_gbuf_image = allocator->create_image(gbuf_create_info);
        ping_pong_res[i].gbuf_ping_pong =
            allocator->create_texture(tmp_gbuf_image, tmp_gbuf_image->make_view_create_info(),
                                      allocator->get_sampler_pool()->nearest_mirrored_repeat());
    }

    {
        ShaderCompileContextHandle compilation_session_desc = ShaderCompileContext::create(context);
        compilation_session_desc->set_preprocessor_macro("FILTER_TYPE",
                                                         std::to_string(filter_type));
        if (kaleidoscope) {
            compilation_session_desc->set_preprocessor_macro("KALEIDOSCOPE", "1");
            if (kaleidoscope_use_shmem) {
                compilation_session_desc->set_preprocessor_macro("KALEIDOSCOPE_USE_SHMEM", "1");
            }
        }
        compilation_session_desc->add_search_path("merian-graph/nodes/svgf");

        filter_module =
            SlangProgramEntryPoint::create(compilation_session_desc, "svgf_filter.slang").get();
        variance_estimate_module =
            SlangProgramEntryPoint::create(compilation_session_desc, "svgf_variance_estimate.slang")
                .get();
        taa_module =
            SlangProgramEntryPoint::create(compilation_session_desc, "svgf_taa.slang").get();

        {
            auto spec_builder = SpecializationInfoBuilder();
            spec_builder.add_entry(variance_estimate_local_size, variance_estimate_local_size,
                                   svgf_iterations);
            SpecializationInfoHandle variance_estimate_spec = spec_builder.build();
            variance_estimate = ComputePipeline::create(
                variance_estimate_module->get_pipeline_layout(context),
                variance_estimate_module->specialize(variance_estimate_spec));
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
                filters[i] = ComputePipeline::create(filter_module->get_pipeline_layout(context),
                                                     filter_module->specialize(filter_spec));
            }
        }
        {
            auto spec_builder = SpecializationInfoBuilder();
            spec_builder.add_entry(taa_local_size, taa_local_size, taa_debug, taa_filter_prev,
                                   taa_clamping, taa_mv_sampling, enable_mv, taa_modulate_albedo);
            SpecializationInfoHandle taa_spec = spec_builder.build();
            taa = ComputePipeline::create(taa_module->get_pipeline_layout(context),
                                          taa_module->specialize(taa_spec));
        }
    }

    // Globals objects; internal ping-pong resources are static per connect and written once,
    // graph io is bound per frame in process().
    variance_estimate_globals =
        variance_estimate_module->create_global_shader_object(context, allocator);
    {
        // variance estimate writes into ping_pong_res[0]
        ShaderCursor cursor = variance_estimate_globals->get_cursor();
        cursor["img_filter_in"].write(ping_pong_res[0].ping_pong->get_view(),
                                      vk::ImageLayout::eGeneral);
        cursor["img_gbuf_in"].write(ping_pong_res[0].gbuf_ping_pong->get_view(),
                                    vk::ImageLayout::eGeneral);
    }

    for (int d = 0; d < 2; d++) {
        filter_globals[d] = filter_module->create_global_shader_object(context, allocator);
        ShaderCursor cursor = filter_globals[d]->get_cursor();
        cursor["img_filter_in"].write(ping_pong_res[d].ping_pong,
                                      vk::ImageLayout::eShaderReadOnlyOptimal);
        cursor["img_filter_out"].write(ping_pong_res[d ^ 1].ping_pong->get_view(),
                                       vk::ImageLayout::eGeneral);
        cursor["img_gbuf_in"].write(ping_pong_res[d].gbuf_ping_pong,
                                    vk::ImageLayout::eShaderReadOnlyOptimal);
        cursor["img_gbuf_out"].write(ping_pong_res[d ^ 1].gbuf_ping_pong->get_view(),
                                     vk::ImageLayout::eGeneral);
    }

    taa_globals = taa_module->create_global_shader_object(context, allocator);
    {
        // after the filter loop the last written texture is ping_pong_res[svgf_iterations & 1]
        ShaderCursor cursor = taa_globals->get_cursor();
        cursor["img_filter_result"].write(ping_pong_res[svgf_iterations & 1].ping_pong,
                                          vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    return {};
}

void SVGF::process(GraphRun& run, const NodeIO& io) {
    const CommandBufferHandle& cmd = run.get_cmd();

    // graph resources can change every iteration (ring, delay)
    {
        ShaderCursor cursor = variance_estimate_globals->get_cursor();
        io.bind(cursor);
    }
    for (int d = 0; d < 2; d++) {
        ShaderCursor cursor = filter_globals[d]->get_cursor();
        io.bind(cursor);
    }
    {
        ShaderCursor cursor = taa_globals->get_cursor();
        io.bind(cursor);
    }

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
        variance_estimate_module->bind_global(variance_estimate_globals, cmd, variance_estimate,
                                              run.get_shader_object_allocator());
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

        // run filter; filter_globals[i & 1] reads ping_pong_res[i & 1], writes write_res
        cmd->bind(filters[i]);
        filter_module->bind_global(filter_globals[i & 1], cmd, filters[i],
                                   run.get_shader_object_allocator());
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
    }

    // TAA
    {
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "taa");
        cmd->bind(taa);
        taa_module->bind_global(taa_globals, cmd, taa, run.get_shader_object_allocator());
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
    float angle = std::acos(variance_estimate_pc.normal_reject_cos);
    config.config_angle("normal reject", angle, "Reject points with farther apart", 0, 180);
    variance_estimate_pc.normal_reject_cos = std::cos(angle);
    config.config_float("depth accept", variance_estimate_pc.depth_accept, "More means more reuse");

    config.st_separate("Filter");
    needs_rebuild |= config.config_int("SVGF iterations", svgf_iterations,
                                       "0 disables SVGF completely (TAA-only mode)", 0, 10);
    config.config_float("filter depth", filter_pc.param_z, "more means more blur");
    angle = std::acos(filter_pc.param_n);
    config.config_angle("filter normals", angle, "Reject with normals farther apart", 0, 180);
    filter_pc.param_n = std::cos(angle);
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
        "TAA alpha", taa_pc.blend_alpha,
        "Blend factor for the final image and the previous image. More means more reuse.", 0.01f,
        0.0f, 1.0f);

    needs_rebuild |=
        config.config_bool("enable motion vectors", enable_mv, "uses motion vectors if connected.");
    needs_rebuild |= config.config_bool(
        "modulate albedo", taa_modulate_albedo,
        "Re-modulate the gbuffer albedo the renderer demodulated out. Disable if the renderer's "
        "'demodulate albedo' is off.");
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

} // namespace merian
