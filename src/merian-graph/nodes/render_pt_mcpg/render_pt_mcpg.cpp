#include "merian-graph/nodes/render_pt_mcpg/render_pt_mcpg.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"

#include <fmt/format.h>

namespace merian {

// DistanceMCVertex in mc_distance.slang: MAX_DISTANCE_MC_VERTEX_STATE_COUNT states of
// { float sum_w; uint N; float2 moments; }.
static constexpr vk::DeviceSize DISTANCE_MC_VERTEX_SIZE = 10 * 16;

RenderMCPG::RenderMCPG() = default;

DeviceSupportInfo RenderMCPG::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_composition(HashedIrradianceCache::query_device_support_composition());
    composition->add_composition(MCPG::query_device_support_composition());
    composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/mcpg_common.slang");
    composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/mc_distance.slang");
    composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/render_pt_mcpg.slang",
                                      true);
    composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/volume.slang", true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayQuery", "rayTracingPipeline"}, {}) &
           program.get()->query_device_support(query_info);
}

void RenderMCPG::initialize(const ContextHandle& context,
                            const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    const bool split_hash_payload_storage = !context->get_device()->get_physical_device()->is_amd();
    mc_split_hash_payload_storage = split_hash_payload_storage;
    lc_split_hash_payload_storage = split_hash_payload_storage;
    irr_cache = std::make_shared<HashedIrradianceCache>(compile_context, allocator, lc_buffer_size,
                                                        lc_probe_count, lc_stochastic_interpolation,
                                                        lc_split_hash_payload_storage);
    mcpg = std::make_shared<MCPG>(compile_context, allocator, mc_adaptive_buffer_size,
                                  mc_split_hash_payload_storage);

    use_raygen = !context->get_device()->get_physical_device()->is_amd();
}

std::vector<InputConnectorDescriptor> RenderMCPG::describe_inputs() {
    return {{"scene", con_scene},
            {"gbuffer", con_gbuffer, ConnectorAccess::ray_tracing_read},
            {"prev_volume_depth", con_prev_volume_depth, ConnectorAccess::compute_read, 1, true}};
}

uint32_t RenderMCPG::distance_mc_vertex_count() const {
    // one vertex per distance_mc_grid_width pixels, plus a border for the jittered lookup
    return (extent.width / distance_mc_grid_width + 2) *
           (extent.height / distance_mc_grid_width + 2);
}

std::vector<OutputConnectorDescriptor> RenderMCPG::describe_outputs(const NodeIOLayout& io_layout) {
    extent = io_layout[con_gbuffer]->get_create_info().extent;
    con_irradiance = ManagedVkImageOut::create(irradiance_format, extent);
    con_debug = ManagedVkImageOut::create(vk::Format::eR16G16B16A16Sfloat, extent);
    con_volume = ManagedVkImageOut::create(irradiance_format, extent);
    con_volume_depth = ManagedVkImageOut::create(volume_depth_format, extent);
    con_volume_mv = ManagedVkImageOut::create(vk::Format::eR16G16Sfloat, extent);

    distance_mc = resource_allocator->create_buffer(
        vk::DeviceSize(distance_mc_vertex_count()) * DISTANCE_MC_VERTEX_SIZE,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::NONE, "RenderMCPG::distance_mc");

    return {{"irradiance", con_irradiance, ConnectorAccess::ray_tracing_write},
            {"debug", con_debug, ConnectorAccess::ray_tracing_write},
            {"volume", con_volume, ConnectorAccess::compute_write},
            {"volume_depth", con_volume_depth, ConnectorAccess::compute_write},
            {"volume_mv", con_volume_mv, ConnectorAccess::compute_read_write}};
}

RenderMCPG::NodeStatusFlags
RenderMCPG::on_connected(const NodeIOLayout& io_layout,
                         const NodeIO& io,
                         [[maybe_unused]] const NodeConnectionInfo& info,
                         [[maybe_unused]] Submission& submission) {

    // force the program graph to be rewired next process()
    composition = nullptr;

    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            if (composition) {
                if (std::any_cast<bool>(force)) {
                    composition->force_reload();
                } else {
                    composition->reload(compile_context->get_search_path_file_loader());
                }
            }
            return true;
        });

    if (const SceneHandle& scene = io[con_scene]; scene && scene->is_ready()) {
        ensure_pipeline(scene);
    }

    return {};
}

void RenderMCPG::ensure_pipeline(const SceneHandle& scene) {
    if (composition) {
        return;
    }

    composition = SlangComposition::create();
    composition->add_composition(scene->get_composition());
    composition->add_composition(irr_cache->get_composition());
    composition->add_composition(mcpg->get_composition());
    composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/mcpg_common.slang");
    composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/mc_distance.slang");
    composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/render_pt_mcpg.slang",
                                      true);
    composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/volume.slang", true);
    update_render_constants();

    program = SlangProgram::create(compile_context, composition);
    entry_point =
        SlangProgramEntryPoint::create(program, use_raygen ? "main_rgen" : "main_compute");

    if (use_raygen) {
        pipeline = Versioned<Pipeline>([this] {
            const auto ep = entry_point.get();
            return RayTracingPipelineBuilder()
                .add_raygen_group(ep->specialize())
                .build(ep->get_pipeline_layout(context));
        });
        pipeline.depends_on(entry_point);

        sbt = Versioned<ShaderBindingTable>([this] {
            return ShaderBindingTable::create(
                std::dynamic_pointer_cast<RayTracingPipeline>(pipeline.get()), resource_allocator);
        });
        sbt.depends_on(pipeline);
    } else {
        pipeline = Versioned<Pipeline>([this] {
            const auto ep = entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        pipeline.depends_on(entry_point);
    }

    params = Versioned<ShaderObject>([this] {
        return entry_point->create_shader_object_for_parameter(context, "params",
                                                               resource_allocator);
    });
    params.depends_on(entry_point);

    const auto build_volume_pass = [this](VolumePass& pass, const std::string& name) {
        pass.entry_point = SlangProgramEntryPoint::create(program, name);
        pass.pipeline = Versioned<Pipeline>([&pass, this] {
            const auto ep = pass.entry_point.get();
            return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
        });
        pass.pipeline.depends_on(pass.entry_point);
        pass.params = Versioned<ShaderObject>([&pass, this] {
            return pass.entry_point->create_shader_object_for_parameter(context, "params",
                                                                        resource_allocator);
        });
        pass.params.depends_on(pass.entry_point);
    };
    build_volume_pass(volume, "volume_main");
    build_volume_pass(volume_mv_init, "volume_mv_init");
    build_volume_pass(volume_project, "volume_forward_project");
}

[[nodiscard]] RenderMCPG::NodeStatusFlags
RenderMCPG::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const auto& cmd = submission.get_cmd();
    const auto& scene = io[con_scene];
    const auto gbuf = io[con_gbuffer];
    if (!scene || !scene->is_ready())
        return {};

    if (max_path_length != emitted_max_path_length) {
        emitted_max_path_length = max_path_length;
        io.send_event("bounces_changed");
    }

    ensure_pipeline(scene);

    const ShaderObjectAllocatorHandle& obj_allocator = info.get_shader_object_allocator();

    const auto ep = entry_point.get();
    const auto pipe = pipeline.get();
    const auto params_obj = params.get();

    // Reset the persistent guiding state on the first frame of a run.
    if (info.get_iteration() == 0) {
        irr_cache->reset(cmd);
        mcpg->reset(cmd);
        cmd->fill(distance_mc);
        cmd->barrier(distance_mc->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite));
    }

    auto cursor = params_obj->get_cursor();
    cursor["gbuffer"] = gbuf.r();
    cursor["irradiance"] = io[con_irradiance].get_texture();
    if (auto debug = cursor["debug"]; debug.is_valid())
        debug = io[con_debug].get_texture();
    irr_cache->write_to(cursor["irr_cache"]);
    mcpg->write_to(cursor["mcpg"]);

    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
    ep->bind("params", params_obj, cmd, pipe, obj_allocator);

    if (use_raygen) {
        cmd->trace_rays(sbt.get(), extent);
    } else {
        cmd->dispatch(extent, 8, 8);
    }

    if (io.is_connected(con_volume)) {
        process_volume(io, info, submission, scene, gbuf);
    }

    return {};
}

void RenderMCPG::process_volume(const NodeIO& io,
                                const NodeProcessInfo& info,
                                Submission& submission,
                                const SceneHandle& scene,
                                const ShaderObjectAccess<GBufferObject>& gbuf) {
    const auto& cmd = submission.get_cmd();

    // Nothing scatters without a medium, and the pass would only burn bandwidth writing zeros.
    if (volume_spp <= 0 || scene->get_exterior_medium().is_vacuum()) {
        cmd->clear(io[con_volume], vk::ImageLayout::eGeneral);
        cmd->clear(io[con_volume_depth], vk::ImageLayout::eGeneral);
        cmd->clear(io[con_volume_mv], vk::ImageLayout::eGeneral);
        return;
    }

    const ShaderObjectAllocatorHandle& obj_allocator = info.get_shader_object_allocator();

    // The three passes share the binding struct, but specialization drops what an entry point does
    // not touch, so every field is written conditionally.
    const auto write_binding = [&](const VolumePass& pass) {
        const auto obj = pass.params.get();
        auto cursor = obj->get_cursor();
        const auto write = [&](const char* name, const auto& value) {
            if (auto field = cursor[name]; field.is_valid())
                field = value;
        };
        write("gbuffer", gbuf.r());
        write("volume", io[con_volume].get_texture());
        write("volume_depth", io[con_volume_depth].get_texture());
        write("volume_mv", io[con_volume_mv].get_texture());
        write("distance_mc", distance_mc);
        if (auto field = cursor["prev_volume_depth"]; field.is_valid())
            field = io[con_prev_volume_depth];
        if (auto field = cursor["irr_cache"]; field.is_valid())
            irr_cache->write_to(field);
        if (auto field = cursor["mcpg"]; field.is_valid())
            mcpg->write_to(field);
        return obj;
    };

    const auto dispatch = [&](const VolumePass& pass, const bool bind_scene) {
        const auto ep = pass.entry_point.get();
        const auto pipe = pass.pipeline.get();
        cmd->bind(pipe);
        if (bind_scene)
            ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
        ep->bind("params", write_binding(pass), cmd, pipe, obj_allocator);
        cmd->dispatch(extent, 8, 8);
    };

    const auto barrier_volume_mv = [&] {
        cmd->barrier(io[con_volume_mv]->barrier2(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::PipelineStageFlagBits2::eComputeShader));
    };

    dispatch(volume_mv_init, false);
    barrier_volume_mv();

    // the ping-pong is only valid from the second iteration on
    if (volume_forward_project && io.is_connected(con_prev_volume_depth) &&
        info.get_iteration() != 0) {
        dispatch(volume_project, true);
        barrier_volume_mv();
    }

    dispatch(volume, true);
}

void RenderMCPG::update_render_constants() {
    uint32_t mask = 0u;
    for (uint32_t bit = 0; bit < 8; ++bit) {
        if (mask_enabled[bit])
            mask |= (1u << bit);
    }

    composition->add_module_from_string(
        "render_pt_mcpg_constants",
        fmt::format("namespace merian {{\n"
                    "export static const bool merian_render_emission_on_primary = {};\n"
                    "export static const int merian_render_spp = {};\n"
                    "export static const int merian_render_max_path_length = {};\n"
                    "export static const uint merian_render_instance_mask = {}u;\n"
                    "export static const bool merian_render_demodulate_albedo = {};\n"
                    "}}\n"
                    "export static const bool use_light_cache_tail = {};\n"
                    "export static const bool missing_light_heuristic = {};\n"
                    "export static const int mc_samples = {};\n"
                    "export static const float p_guiding = {:f};\n"
                    "export static const float dir_guide_prior = {:f};\n"
                    "export static const int debug_output_selector = {};\n"
                    "export static const uint lc_buffer_size = {}u;\n"
                    "export static const uint lc_probe_count = {}u;\n"
                    "export static const bool lc_stochastic_interpolation = {};\n"
                    "export static const bool lc_split_storage = {};\n"
                    "export static const uint lc_locality_bits = {}u;\n"
                    "export static const float lc_min_pdf = {:f};\n"
                    "export static const uint mc_adaptive_buffer_size = {}u;\n"
                    "export static const uint mc_probe_count = {}u;\n"
                    "export static const bool mc_split_storage = {};\n"
                    "export static const uint mc_locality_bits = {}u;\n"
                    "export static const int volume_spp = {};\n"
                    "export static const bool volume_use_light_cache = {};\n"
                    "export static const float volume_p_guiding = {:f};\n"
                    "export static const float volume_p_dist_guiding = {:f};\n"
                    "export static const float volume_forward_project_min_z = {:f};\n"
                    "export static const int distance_mc_samples = {};\n"
                    "export static const int distance_mc_grid_width = {};\n"
                    "export static const uint distance_mc_vertex_state_count = {}u;\n",
                    emission_on_primary ? "true" : "false", spp, max_path_length, mask,
                    demodulate_albedo ? "true" : "false", use_light_cache_tail ? "true" : "false",
                    missing_light_heuristic ? "true" : "false", mc_samples,
                    reference_mode ? 0.0f : p_guiding, dir_guide_prior, debug_output_selector,
                    lc_buffer_size, lc_probe_count, lc_stochastic_interpolation ? "true" : "false",
                    lc_split_hash_payload_storage ? "true" : "false", lc_locality_bits, lc_min_pdf,
                    mc_adaptive_buffer_size, mc_probe_count,
                    mc_split_hash_payload_storage ? "true" : "false", mc_locality_bits, volume_spp,
                    volume_use_light_cache ? "true" : "false",
                    reference_mode ? 0.0f : volume_p_guiding,
                    reference_mode ? 0.0f : volume_p_dist_guiding, volume_forward_project_min_z,
                    distance_mc_samples, distance_mc_grid_width, distance_mc_vertex_state_count));
}

RenderMCPG::NodeStatusFlags RenderMCPG::properties(Properties& config) {
    bool needs_reconnect = false;
    bool constants_changed = false;

    config.st_separate("General");
    needs_reconnect |=
        config.config_bool("raygen executor", use_raygen,
                           "Trace from a ray-tracing pipeline (raygen shader) instead of a compute "
                           "shader. Necessary for SER support.");
    constants_changed |= config.config_bool("reference mode", reference_mode,
                                            "Disable guiding (pure BSDF sampling).");
    constants_changed |=
        config.config_bool("emission on primary", emission_on_primary,
                           "Fold primary-hit emission into irradiance (self-contained). "
                           "Otherwise it is the GBuffer emission texture's job.");
    constants_changed |= config.config_bool(
        "demodulate albedo", demodulate_albedo,
        "Divide the primary-hit albedo out of the output so a denoiser can re-modulate after "
        "filtering. Use with 'emission on primary' disabled (emission is albedo-independent).");

    config.st_separate("RT Surface");
    constants_changed |=
        config.config_int("samples per pixel", spp, "Number of paths per pixel.", 1, 16);
    constants_changed |= config.config_int("max path length", max_path_length,
                                           "Maximum number of path segments, including the "
                                           "primary hit.",
                                           1, 16);
    constants_changed |=
        config.config_percent("guiding prob", p_guiding,
                              "Probability to sample the guiding distribution instead of "
                              "the BSDF.");

    config.st_separate("RT Volume");
    constants_changed |= config.config_int("volume samples per pixel", volume_spp,
                                           "Number of single-scattering events per pixel. Only "
                                           "runs while the scene declares an exterior medium.",
                                           0, 16);
    constants_changed |=
        config.config_percent("volume guiding prob", volume_p_guiding,
                              "Probability to sample the guiding distribution instead of the "
                              "phase function.");
    constants_changed |=
        config.config_percent("distance guiding prob", volume_p_dist_guiding,
                              "Probability to sample the scattering distance from the per-pixel "
                              "chain instead of the transmittance.");
    constants_changed |=
        config.config_bool("volume: use LC", volume_use_light_cache,
                           "Query the light cache for non-emitting surfaces behind the "
                           "scattering event.");
    constants_changed |= config.config_int("distance MC samples", distance_mc_samples, "", 0, 30);
    needs_reconnect |= config.config_int("distance MC grid width", distance_mc_grid_width,
                                         "Side length of a distance chain cell, in pixels.", 1, 64);
    constants_changed |=
        config.config_uint("distance MC states per vertex", distance_mc_vertex_state_count,
                           "Number of chains per cell.", 1u, 10u);
    config.config_bool("volume forward project", volume_forward_project,
                       "Reproject last frame's mean scattering distance into the volume motion "
                       "vectors instead of using the surface ones.");
    constants_changed |= config.config_float(
        "volume forward project min z", volume_forward_project_min_z,
        "Below this scattering distance the surface motion vector is kept.", 1.f, 0.f);

    if (config.st_begin_child("mc", "Markov Chain Path Guiding",
                              Properties::ChildFlagBits::DEFAULT_OPEN)) {
        constants_changed |= config.config_percent("ML prior", dir_guide_prior);
        constants_changed |= config.config_int("MC samples", mc_samples, "", 0, 30);
        constants_changed |= config.config_bool(
            "missing light heuristic", missing_light_heuristic,
            "Flood the Markov chains with invalidated states when no light is detected.");
        bool recreate_mcpg = config.config_uint("adaptive grid buf size", mc_adaptive_buffer_size,
                                                "Buffer size backing the hash grid.");
        constants_changed |=
            config.config_uint("MC probe count", mc_probe_count,
                               "Slots probed before evicting (open addressing).", 1u, 32u);
        recreate_mcpg |=
            config.config_bool("split keys/payload", mc_split_hash_payload_storage,
                               "Store hash+stamp separately from the payload (probe-friendly) "
                               "instead of one combined record per slot.");
        constants_changed |= config.config_uint(
            "locality bits", mc_locality_bits,
            "Give each 2^n-wide cell tile a contiguous Morton-ordered slot range so nearby "
            "cells share cache lines (0 = scatter every cell).",
            0u, 5u);
        needs_reconnect |= recreate_mcpg;
        if (mcpg) {
            if (recreate_mcpg) {
                mcpg =
                    std::make_shared<MCPG>(compile_context, resource_allocator,
                                           mc_adaptive_buffer_size, mc_split_hash_payload_storage);
                if (composition)
                    update_render_constants();
            }
            mcpg->properties(config);
        }
        config.st_end_child();
    }

    if (config.st_begin_child("lc", "Light cache", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        constants_changed |= config.config_bool("surf: use LC", use_light_cache_tail,
                                                "Use the light cache for the path tail.");
        bool recreate_cache = false;
        recreate_cache |=
            config.config_uint("LC buffer size", lc_buffer_size,
                               "Number of cache slots backing the hash grid.", 1u, 100000000u);
        recreate_cache |=
            config.config_uint("LC probe count", lc_probe_count,
                               "Slots probed before evicting (open addressing).", 1u, 16u);
        recreate_cache |=
            config.config_bool("LC stochastic interpolation", lc_stochastic_interpolation,
                               "Jitter the grid cell per sample (smoother but noisier) "
                               "instead of snapping to the nearest cell.");
        recreate_cache |=
            config.config_bool("LC split keys/payload", lc_split_hash_payload_storage,
                               "Store hash+stamp separately from the payload (probe-friendly) "
                               "instead of one combined record per slot.");
        constants_changed |= config.config_uint(
            "LC locality bits", lc_locality_bits,
            "Give each 2^n-wide cell tile a contiguous Morton-ordered slot range so nearby "
            "cells share cache lines (0 = scatter every cell).",
            0u, 5u);
        constants_changed |= config.config_float(
            "LC min pdf", lc_min_pdf,
            "Increase to reduce fireflies in the irradiance cache and bias the guiding towards "
            "direct light, especially useful for short maximum path lengths.",
            0.1f, 0.0f);
        // Fail gracefully if compilation fails.
        if (irr_cache) {
            if (recreate_cache) {
                irr_cache = std::make_shared<HashedIrradianceCache>(
                    compile_context, resource_allocator, lc_buffer_size, lc_probe_count,
                    lc_stochastic_interpolation, lc_split_hash_payload_storage);
                if (composition)
                    update_render_constants();
            }
            irr_cache->properties(config);
        }
        config.st_end_child();
    }

    config.st_separate("instance mask");
    for (uint32_t bit = 0; bit < 8; ++bit) {
        constants_changed |= config.config_bool(std::to_string(bit), mask_enabled[bit]);
        if ((bit & 3u) != 3u)
            config.st_no_space();
    }

    config.st_separate("Debug");
    constants_changed |=
        config.config_options("debug output", debug_output_selector,
                              {"irradiance", "moments", "light cache", "mc grid", "mc lod",
                               "mc weight", "mc mean direction", "mc cos", "mc N", "mc mv",
                               "lc normal bin (actual)", "lc normal bin (selected)"});

    if (constants_changed && composition) {
        update_render_constants();
    }

    config.st_separate();
    needs_reconnect |=
        config.config_enum("irradiance format", irradiance_format, Properties::OptionsStyle::COMBO);

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
