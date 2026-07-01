#include "merian-graph/nodes/render_pt_mcpg/render_pt_mcpg.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"

#include <fmt/format.h>

#include <random>

namespace merian {

RenderMCPG::RenderMCPG() = default;

DeviceSupportInfo RenderMCPG::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_composition(HashedIrradianceCache::query_device_support_composition());
    composition->add_composition(MCPG::query_device_support_composition());
    composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/render_pt_mcpg.slang",
                                      true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline"}, {"rayQuery"}) &
           program.get()->query_device_support(query_info);
}

void RenderMCPG::initialize(const ContextHandle& context,
                            const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();
    irr_cache = std::make_shared<HashedIrradianceCache>(
        compile_context, allocator, lc_buffer_size, lc_probe_count, lc_stochastic_interpolation);
    mcpg = std::make_shared<MCPG>(compile_context, allocator, mc_adaptive_buffer_size);
}

std::vector<InputConnectorDescriptor> RenderMCPG::describe_inputs() {
    return {{"scene", con_scene}, {"gbuffer", con_gbuffer}};
}

std::vector<OutputConnectorDescriptor>
RenderMCPG::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    con_irradiance = ManagedVkImageOut::compute_write(vk::Format::eR32G32B32A32Sfloat, extent);
    con_debug = ManagedVkImageOut::compute_write(vk::Format::eR16G16B16A16Sfloat, extent);
    return {{"irradiance", con_irradiance}, {"debug", con_debug}};
}

RenderMCPG::NodeStatusFlags
RenderMCPG::on_connected(const NodeIOLayout& io_layout,
                         [[maybe_unused]] const DescriptorSetLayoutHandle& descriptor_set_layout) {

    // force the program graph to be rewired next process()
    composition = nullptr;
    obj_allocator = nullptr;

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

    return {};
}

void RenderMCPG::process(GraphRun& run,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) {
    const auto& cmd = run.get_cmd();
    const auto& scene = io[con_scene];
    const auto& gbuf = io[con_gbuffer];
    if (!scene || !gbuf || !scene->is_ready())
        return;

    if (max_path_length != emitted_max_path_length) {
        emitted_max_path_length = max_path_length;
        io.send_event("bounces_changed");
    }

    if (!composition) {
        if (randomize_seed) {
            std::random_device dev;
            std::mt19937 rng(dev());
            seed = std::uniform_int_distribution<uint32_t>{}(rng);
        }

        composition = SlangComposition::create();
        composition->add_composition(scene->get_composition());
        composition->add_composition(irr_cache->get_composition());
        composition->add_composition(mcpg->get_composition());
        composition->add_module_from_path("merian-graph/nodes/render_pt_mcpg/render_pt_mcpg.slang",
                                          true);
        update_render_constants();

        program = SlangProgram::create(compile_context, composition);
        entry_point = SlangProgramEntryPoint::create(program, "main");

        pipeline = Versioned<RayTracingPipeline>([this] {
            const auto ep = entry_point.get();
            return RayTracingPipelineBuilder()
                .add_raygen_group(ep->specialize())
                .build(ep->get_pipeline_layout(context));
        });
        pipeline.depends_on(entry_point);

        sbt = Versioned<ShaderBindingTable>(
            [this] { return ShaderBindingTable::create(pipeline.get(), resource_allocator); });
        sbt.depends_on(pipeline);

        params = Versioned<ShaderObject>([this] {
            return entry_point->create_shader_object_for_parameter(context, "params",
                                                                   resource_allocator);
        });
        params.depends_on(entry_point);

        obj_allocator = std::make_shared<FrameCachingShaderObjectAllocator>(
            resource_allocator, run.get_iterations_in_flight());
    }

    obj_allocator->set_iteration(run.get_in_flight_index());

    const auto ep = entry_point.get();
    const auto pipe = pipeline.get();
    const auto params_obj = params.get();

    // Reset the persistent guiding state on the first frame of a run.
    if (run.get_iteration() == 0) {
        irr_cache->reset(cmd);
        mcpg->reset(cmd);
    }

    auto cursor = params_obj->get_cursor();
    cursor["gbuffer"] = gbuf->get_shader_object();
    cursor["irradiance"] = io[con_irradiance].get_texture();
    if (auto debug = cursor["debug"]; debug.is_valid())
        debug = io[con_debug].get_texture();
    irr_cache->write_to(cursor["irr_cache"]);
    mcpg->write_to(cursor["mcpg"]);

    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
    ep->bind("params", params_obj, cmd, pipe, obj_allocator);

    cmd->trace_rays(sbt.get(), extent);
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
                    "}}\n"
                    "export static const bool reference_mode = {};\n"
                    "export static const bool use_light_cache_tail = {};\n"
                    "export static const bool missing_light_heuristic = {};\n"
                    "export static const int mc_samples = {};\n"
                    "export static const float mc_samples_adaptive_prob = {:f};\n"
                    "export static const float p_guiding = {:f};\n"
                    "export static const float dir_guide_prior = {:f};\n"
                    "export static const uint seed = {}u;\n"
                    "export static const int debug_output_selector = {};\n"
                    "export static const uint lc_buffer_size = {}u;\n"
                    "export static const uint lc_probe_count = {}u;\n"
                    "export static const bool lc_stochastic_interpolation = {};\n"
                    "export static const uint lc_normal_bits = {}u;\n"
                    "export static const uint mc_adaptive_buffer_size = {}u;\n"
                    "export static const uint mc_normal_bits = {}u;\n",
                    emission_on_primary ? "true" : "false", spp, max_path_length, mask,
                    (reference_mode || p_guiding == 0.0f) ? "true" : "false",
                    use_light_cache_tail ? "true" : "false",
                    missing_light_heuristic ? "true" : "false", mc_samples,
                    mc_samples_adaptive_prob, p_guiding, dir_guide_prior, seed,
                    debug_output_selector, lc_buffer_size, lc_probe_count,
                    lc_stochastic_interpolation ? "true" : "false", lc_normal_bits,
                    mc_adaptive_buffer_size, mc_normal_bits));
}

RenderMCPG::NodeStatusFlags RenderMCPG::properties(Properties& config) {
    bool needs_reconnect = false;
    bool constants_changed = false;

    config.st_separate("General");
    config.config_bool("randomize seed", randomize_seed,
                       "Randomize the seed on every graph build.");
    if (randomize_seed) {
        config.output_text(fmt::format("seed: {}", seed));
    } else {
        constants_changed |= config.config_uint("seed", seed);
    }
    constants_changed |= config.config_bool("reference mode", reference_mode,
                                            "Disable guiding (pure BSDF sampling).");
    constants_changed |=
        config.config_bool("emission on primary", emission_on_primary,
                           "Fold primary-hit emission into irradiance (self-contained). "
                           "Otherwise it is the GBuffer emission texture's job.");

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

    if (config.st_begin_child("mc", "Markov Chain Path Guiding",
                              Properties::ChildFlagBits::DEFAULT_OPEN)) {
        constants_changed |= config.config_percent("ML prior", dir_guide_prior);
        constants_changed |= config.config_int("MC samples", mc_samples, "", 0, 30);
        constants_changed |= config.config_percent("adaptive grid prob", mc_samples_adaptive_prob);
        constants_changed |= config.config_bool(
            "missing light heuristic", missing_light_heuristic,
            "Flood the Markov chains with invalidated states when no light is detected.");
        constants_changed |= config.config_uint(
            "adaptive grid normal bits", mc_normal_bits,
            "Octahedral normal bins folded into the hash key; neighbouring bins share. 0 ignores "
            "the normal.",
            0u, 16u);
        const bool resize_mcpg =
            config.config_uint("adaptive grid buf size", mc_adaptive_buffer_size,
                               "Buffer size backing the hash grid.");
        needs_reconnect |= resize_mcpg;
        if (mcpg) {
            if (resize_mcpg) {
                mcpg = std::make_shared<MCPG>(compile_context, resource_allocator,
                                              mc_adaptive_buffer_size);
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
        constants_changed |= config.config_uint(
            "LC normal bits", lc_normal_bits,
            "Octahedral normal bins folded into the hash key; neighbouring bins share. 0 ignores "
            "the normal.",
            0u, 16u);
        // Fail gracefully if compilation fails.
        if (irr_cache) {
            if (recreate_cache) {
                irr_cache = std::make_shared<HashedIrradianceCache>(
                    compile_context, resource_allocator, lc_buffer_size, lc_probe_count,
                    lc_stochastic_interpolation);
                if (composition)
                    update_render_constants();
            }
            irr_cache->properties(config);
        }
        config.st_end_child();
    }

    config.st_separate("Resolution");
    needs_reconnect |= config.config_uint("width", &extent.width);
    needs_reconnect |= config.config_uint("height", &extent.height);

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

    if (needs_reconnect) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
