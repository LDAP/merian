#include "merian-graph/nodes/render_restir_cgns/render_restir_cgns.hpp"

#include "merian-graph/nodes/render_restir_cgns/render_restir_cgns.slangh"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/utils/small_vector.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <fmt/format.h>

namespace merian {

namespace {
constexpr const char* SHADER_MODULE =
    "merian-graph/nodes/render_restir_cgns/render_restir_cgns.slang";
constexpr std::array<const char*, RenderRestirCGNS::PassCount> PASS_ENTRY_POINTS = {
    "initial", "temporal", "select_neighbors", "spatial"};
} // namespace

RenderRestirCGNS::RenderRestirCGNS() = default;

DeviceSupportInfo RenderRestirCGNS::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_module_from_path(SHADER_MODULE, true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline"}, {"rayQuery"}) &
           program.get()->query_device_support(query_info);
}

void RenderRestirCGNS::initialize(const ContextHandle& context,
                                  const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    use_raygen = !context->get_device()->get_physical_device()->is_amd();
}

vk::BufferCreateInfo RenderRestirCGNS::reservoir_buffer_create_info() const {
    return vk::BufferCreateInfo{
        {},
        vk::DeviceSize(extent.width) * extent.height * sizeof(CgnsReservoir),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eTransferDst};
}

vk::BufferCreateInfo RenderRestirCGNS::reconnection_buffer_create_info() const {
    return vk::BufferCreateInfo{
        {},
        vk::DeviceSize(extent.width) * extent.height * sizeof(CgnsReconnection),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eTransferDst};
}

vk::BufferCreateInfo RenderRestirCGNS::neighbor_buffer_create_info() const {
    return vk::BufferCreateInfo{{},
                                vk::DeviceSize(extent.width) * extent.height * neighbor_count * 8,
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eShaderDeviceAddress};
}

vk::BufferCreateInfo RenderRestirCGNS::splat_buffer_create_info() const {
    return vk::BufferCreateInfo{{},
                                vk::DeviceSize(extent.width) * extent.height * splat_capacity * 4,
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eShaderDeviceAddress};
}

vk::BufferCreateInfo RenderRestirCGNS::splat_count_buffer_create_info() const {
    return vk::BufferCreateInfo{{},
                                vk::DeviceSize(extent.width) * extent.height * 4,
                                vk::BufferUsageFlagBits::eStorageBuffer |
                                    vk::BufferUsageFlagBits::eTransferDst};
}

void RenderRestirCGNS::update_render_constants() {
    uint32_t mask = 0u;
    for (uint32_t bit = 0; bit < 8; ++bit) {
        if (mask_enabled[bit])
            mask |= (1u << bit);
    }

    composition->add_module_from_string(
        "render_restir_cgns_constants",
        fmt::format("namespace merian {{\n"
                    "export static const bool merian_cgns_emission_on_primary = {};\n"
                    "export static const bool merian_cgns_russian_roulette = {};\n"
                    "export static const bool merian_cgns_demodulate_albedo = {};\n"
                    "export static const int merian_cgns_spp = {};\n"
                    "export static const int merian_cgns_max_path_length = {};\n"
                    "export static const uint merian_cgns_instance_mask = {}u;\n"
                    "export static const bool merian_cgns_confidence_temporal = {};\n"
                    "export static const bool merian_cgns_confidence_spatial = {};\n"
                    "export static const float merian_cgns_confidence_cap = {:f};\n"
                    "export static const bool merian_cgns_geometry_rejection = {};\n"
                    "export static const float merian_cgns_reject_normal = {:f};\n"
                    "export static const float merian_cgns_reject_depth = {:f};\n"
                    "export static const int merian_cgns_neighbor_count = {};\n"
                    "export static const int merian_cgns_candidates = {};\n"
                    "export static const bool merian_cgns_early_stopping = {};\n"
                    "export static const int merian_cgns_temporal_mode = {};\n"
                    "export static const int merian_cgns_splat_capacity = {};\n"
                    "}}",
                    emission_on_primary ? "true" : "false", russian_roulette ? "true" : "false",
                    demodulate_albedo ? "true" : "false", spp, max_path_length, mask,
                    confidence_temporal ? "true" : "false", confidence_spatial ? "true" : "false",
                    confidence_cap, geometry_rejection ? "true" : "false", reject_normal,
                    reject_depth, neighbor_count, candidates, early_stopping ? "true" : "false",
                    static_cast<int>(temporal_mode), splat_capacity));
}

std::vector<InputConnectorDescriptor> RenderRestirCGNS::describe_inputs() {
    return {{.name = "scene", .connector = con_scene},
            {.name = "gbuffer", .connector = con_gbuffer, .access = ConnectorAccess::compute_read},
            {.name = "prev_gbuffer",
             .connector = con_prev_gbuffer,
             .access = ConnectorAccess::compute_read,
             .delay = 1},
            {.name = "prev_reservoirs",
             .connector = con_prev_reservoirs,
             .access = ConnectorAccess::compute_read,
             .delay = 1},
            {.name = "prev_reconnection",
             .connector = con_prev_reconnection,
             .access = ConnectorAccess::compute_read,
             .delay = 1}};
}

std::vector<OutputConnectorDescriptor>
RenderRestirCGNS::describe_outputs(const NodeIOLayout& io_layout) {
    extent = io_layout[con_gbuffer]->get_create_info().extent;
    con_irradiance = ManagedVkImageOut::create(irradiance_format, extent);
    con_reservoirs = ManagedVkBufferOut::create(reservoir_buffer_create_info());
    con_reconnection = ManagedVkBufferOut::create(reconnection_buffer_create_info());
    return {{.name = "irradiance",
             .connector = con_irradiance,
             .access = ConnectorAccess::compute_write},
            {.name = "reservoirs",
             .connector = con_reservoirs,
             .access = ConnectorAccess::compute_read_write},
            {.name = "reconnection",
             .connector = con_reconnection,
             .access = ConnectorAccess::compute_read_write}};
}

RenderRestirCGNS::NodeStatusFlags
RenderRestirCGNS::on_connected(const NodeIOLayout& io_layout,
                               const NodeIO& io,
                               [[maybe_unused]] const NodeConnectionInfo& info,
                               Submission& submission) {
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

    pong_reservoirs = resource_allocator->create_buffer(
        reservoir_buffer_create_info(), MemoryMappingType::NONE, "ReSTIR CGNS reservoirs");
    pong_reconnection = resource_allocator->create_buffer(
        reconnection_buffer_create_info(), MemoryMappingType::NONE, "ReSTIR CGNS reconnection");
    neighbors = resource_allocator->create_buffer(neighbor_buffer_create_info(),
                                                  MemoryMappingType::NONE, "ReSTIR CGNS neighbors");
    splats = resource_allocator->create_buffer(splat_buffer_create_info(), MemoryMappingType::NONE,
                                               "ReSTIR CGNS splats");
    splat_counts = resource_allocator->create_buffer(
        splat_count_buffer_create_info(), MemoryMappingType::NONE, "ReSTIR CGNS splat counts");

    // the first temporal pass resamples from reservoirs that no pass has written yet
    submission.get_cmd()->fill(io[con_reservoirs]);
    submission.get_cmd()->fill(io[con_reconnection]);

    if (const SceneHandle& scene = io[con_scene]; scene && scene->is_ready()) {
        ensure_pipeline(scene);
    }

    return {};
}

void RenderRestirCGNS::ensure_pipeline(const SceneHandle& scene) {
    if (composition) {
        return;
    }

    composition = SlangComposition::create();
    composition->add_composition(scene->get_composition());
    composition->add_module_from_path(SHADER_MODULE, true);
    update_render_constants();
    program = SlangProgram::create(compile_context, composition);

    for (uint32_t p = 0; p < PassCount; p++) {
        // The candidate trace is the only pass long and divergent enough to pay for a ray tracing
        // pipeline; the resampling passes stay compute.
        const bool raygen = use_raygen && p == Initial;
        entry_points[p] = SlangProgramEntryPoint::create(
            program, raygen ? fmt::format("{}_rt", PASS_ENTRY_POINTS[p])
                            : std::string(PASS_ENTRY_POINTS[p]));

        if (raygen) {
            pipelines[p] = Versioned<Pipeline>([this, p] {
                const auto ep = entry_points[p].get();
                return RayTracingPipelineBuilder()
                    .add_raygen_group(ep->specialize())
                    .build(ep->get_pipeline_layout(context));
            });
            pipelines[p].depends_on(entry_points[p]);

            initial_sbt = Versioned<ShaderBindingTable>([this] {
                return ShaderBindingTable::create(
                    std::dynamic_pointer_cast<RayTracingPipeline>(pipelines[Initial].get()),
                    resource_allocator);
            });
            initial_sbt.depends_on(pipelines[p]);
        } else {
            pipelines[p] = Versioned<Pipeline>([this, p] {
                const auto ep = entry_points[p].get();
                return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
            });
            pipelines[p].depends_on(entry_points[p]);
        }

        params[p] = Versioned<ShaderObject>([this, p] {
            return entry_points[p]->create_shader_object_for_parameter(context, "params",
                                                                       resource_allocator);
        });
        params[p].depends_on(entry_points[p]);
    }
}

[[nodiscard]] RenderRestirCGNS::NodeStatusFlags
RenderRestirCGNS::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const auto& cmd = submission.get_cmd();
    const auto& scene = io[con_scene];
    if (!scene || !scene->is_ready())
        return {};

    if (max_path_length != emitted_max_path_length) {
        emitted_max_path_length = max_path_length;
        io.send_event("bounces_changed");
    }

    ensure_pipeline(scene);
    const ShaderObjectAllocatorHandle& obj_allocator = info.get_shader_object_allocator();

    struct Set {
        BufferHandle reservoirs;
        BufferHandle reconnection;
    };
    const Set out{io[con_reservoirs], io[con_reconnection]};
    const Set scratch{pong_reservoirs, pong_reconnection};
    // an odd number of spatial rounds starts in the scratch set so the last one lands on the output
    Set current = (spatial_rounds % 2) != 0 ? scratch : out;

    CgnsPushConstant pc{};
    pc.reservoirs_prev = io[con_prev_reservoirs]->get_device_address();
    pc.reconnection_prev = io[con_prev_reconnection]->get_device_address();
    pc.neighbors = neighbors->get_device_address();
    pc.splats = splats->get_device_address();
    pc.frame = static_cast<uint32_t>(info.get_iteration());
    pc.seed = seed;
    pc.spatial_radius = spatial_radius;
    pc.scale_solid_angle = scale_solid_angle;
    pc.normal_beta = normal_beta;
    pc.early_stop_cutoff = early_stop_cutoff;

    const auto run = [&](const Pass p, const Set& in, const Set& write, const uint32_t flags) {
        const auto ep = entry_points[p].get();
        const auto pipe = pipelines[p].get();
        const auto obj = params[p].get();

        auto cursor = obj->get_cursor();
        cursor["gbuffer"] = io[con_gbuffer].r();
        cursor["prev_gbuffer"] = io[con_prev_gbuffer].r();
        cursor["irradiance"] = io[con_irradiance].get_texture();
        cursor["splat_counts"] = splat_counts;

        pc.pass = static_cast<uint32_t>(p);
        pc.flags = flags;
        pc.reservoirs_in = in.reservoirs->get_device_address();
        pc.reconnection_in = in.reconnection->get_device_address();
        pc.reservoirs_out = write.reservoirs->get_device_address();
        pc.reconnection_out = write.reconnection->get_device_address();

        cmd->bind(pipe);
        ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
        ep->bind("params", obj, cmd, pipe, obj_allocator);
        cmd->push_constant(pipe, pc);
        if (use_raygen && p == Initial) {
            cmd->trace_rays(initial_sbt.get(), extent);
        } else {
            cmd->dispatch(extent, 8, 8);
        }
    };

    const auto sync = [&](const std::initializer_list<BufferHandle> buffers) {
        SmallVector<vk::BufferMemoryBarrier, 3> barriers;
        for (const BufferHandle& buffer : buffers) {
            barriers.push_back(buffer->buffer_barrier(
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite));
        }
        vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eComputeShader;
        if (use_raygen) {
            stage |= vk::PipelineStageFlagBits::eRayTracingShaderKHR;
        }
        cmd->barrier(stage, stage,
                     vk::ArrayProxy<const vk::BufferMemoryBarrier>(
                         static_cast<uint32_t>(barriers.size()), barriers.data()));
    };

    // whichever dispatch runs last also writes the image, so no separate resolve pass is needed
    const bool temporal = temporal_enable && info.get_iteration() > 0;
    const uint32_t last = spatial_rounds == 0 ? CgnsPassResolves : 0u;
    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "initial candidates");
        run(Initial, current, current, temporal ? 0u : last);
    }

    if (temporal) {
        if (temporal_mode != TemporalMode::Gather) {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "splat");
            cmd->fill(splat_counts, 0);
            cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                         vk::PipelineStageFlagBits::eComputeShader,
                         splat_counts->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite));
            run(Temporal, current, current, CgnsPassSplats);
            sync({splats, splat_counts});
        }
        sync({current.reservoirs, current.reconnection});
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "temporal");
        run(Temporal, current, current, last);
    }

    for (int32_t round = 0; round < spatial_rounds; round++) {
        const Set& write = current.reservoirs == out.reservoirs ? scratch : out;
        pc.spatial_round = static_cast<uint32_t>(round);
        sync({current.reservoirs, current.reconnection, neighbors});
        {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "select neighbors");
            run(SelectNeighbors, current, write, 0);
        }
        sync({neighbors});
        {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "spatial");
            run(Spatial, current, write, round + 1 == spatial_rounds ? CgnsPassResolves : 0u);
        }
        current = write;
    }

    return {};
}

RenderRestirCGNS::NodeStatusFlags RenderRestirCGNS::properties(Properties& config) {
    bool needs_reconnect = false;
    bool constants_changed = false;

    config.st_separate("initial candidates");
    constants_changed |=
        config.config_int("samples per pixel", spp, "Paths traced per pixel.", 1, 16);
    constants_changed |=
        config.config_int("max path length", max_path_length,
                          "Maximum number of path segments, including the primary hit.", 2, 16);
    constants_changed |= config.config_bool(
        "russian roulette", russian_roulette,
        "Terminate low-throughput paths after the vertex a shift reconnects at.");
    constants_changed |=
        config.config_bool("emission on primary", emission_on_primary,
                           "Fold primary-hit emission into irradiance (self-contained). "
                           "Otherwise it is the GBuffer emission texture's job.");
    config.config_uint("seed", seed, "Base seed for the per-pixel RNG.");
    constants_changed |= config.config_bool(
        "demodulate albedo", demodulate_albedo,
        "Divide the primary-hit albedo out of the output so a denoiser can re-modulate after "
        "filtering. Use with 'emission on primary' disabled (emission is albedo-independent).");

    config.st_separate("temporal reuse");
    int32_t mode = static_cast<int32_t>(temporal_mode);
    constants_changed |= config.config_options(
        "reprojection", mode, {"gather", "splat", "splat, gather fallback"},
        Properties::OptionsStyle::COMBO,
        "How the previous frame's reservoirs reach a pixel. 'gather' pulls the one this pixel's "
        "motion vector points at; 'splat' pushes every reservoir onto the pixel its own primary "
        "vertex projects to.");
    temporal_mode = static_cast<TemporalMode>(mode);
    if (temporal_mode != TemporalMode::Gather) {
        needs_reconnect |=
            config.config_int("splat capacity", splat_capacity,
                              "Reservoirs a pixel can receive. Any past that are dropped.", 1, 8);
    }
    config.config_bool("enable temporal reuse", temporal_enable);
    constants_changed |= config.config_bool(
        "confidence weights##temporal", confidence_temporal,
        "Weigh the two domains by how many samples each stands for instead of equally.");
    constants_changed |= config.config_float(
        "history cap", confidence_cap,
        "Samples a reservoir may stand for. Longer histories converge faster but correlate "
        "neighbouring pixels, which the resampling weights assume they are not.",
        1.f);

    config.st_separate("spatial reuse");
    config.config_int("rounds", spatial_rounds,
                      "Spatial resampling passes; each one re-selects its neighbors.", 0, 4);
    config.config_float("radius", spatial_radius, "Pixel radius the candidate disk covers.", 0.f);
    constants_changed |= config.config_bool("confidence weights##spatial", confidence_spatial);
    constants_changed |= config.config_bool(
        "geometry rejection", geometry_rejection,
        "Drop selected neighbors whose normal or depth diverges too far from the pixel's.");
    if (geometry_rejection) {
        float angle = std::acos(reject_normal);
        constants_changed |= config.config_angle(
            "normal threshold", angle, "Reject neighbors with normals farther apart.", 0, 180);
        reject_normal = std::cos(angle);
        constants_changed |= config.config_percent(
            "depth threshold", reject_depth,
            "Reject neighbors whose camera distance differs by more than this fraction.");
    }

    config.st_separate("CGNS neighbor selection");
    needs_reconnect |= config.config_int(
        "neighbors", neighbor_count,
        "M: neighbors resampled per pixel. One uses A-Chao, more use A-ES weighted reservoir "
        "sampling.",
        1, 5);
    constants_changed |=
        config.config_int("candidates", candidates,
                          "K: candidates scored per pixel before M of them are kept.", 1, 128);
    config.config_float("solid angle", scale_solid_angle,
                        "Solid angle of the characteristic search disk, in steradian. Sets the "
                        "world-space falloff of the position term, so the heuristic does not "
                        "depend on scene scale.",
                        0.f);
    config.config_float("normal exponent", normal_beta,
                        "Exponent on the normal compatibility term.", 0.f);
    constants_changed |= config.config_bool(
        "early stopping", early_stopping,
        "Stop scoring candidates once M of them are compatible enough to be worth keeping.");
    if (early_stopping) {
        config.config_float("early stop cutoff", early_stop_cutoff,
                            "Compatibility score a candidate must reach to stop the search.", 0.f,
                            1.f);
    }

    config.st_separate("instance mask");
    for (uint32_t bit = 0; bit < 8; ++bit) {
        constants_changed |= config.config_bool(std::to_string(bit), mask_enabled[bit]);
        if ((bit & 3u) != 3u)
            config.st_no_space();
    }

    if (constants_changed && composition) {
        update_render_constants();
    }

    config.st_separate();
    needs_reconnect |= config.config_bool(
        "ray tracing pipeline", use_raygen,
        "Trace from a raygen shader instead of a compute shader. The compute path avoids the "
        "ray-tracing pipeline register cap.");
    needs_reconnect |=
        config.config_enum("irradiance format", irradiance_format, Properties::OptionsStyle::COMBO);

    if (needs_reconnect)
        return NEEDS_RECONNECT;
    return {};
}

} // namespace merian
