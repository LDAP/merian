#include "merian-graph/nodes/render_restir_pt/render_restir_pt.hpp"

#include "merian-graph/nodes/render_restir_pt/render_restir_pt.slangh"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/utils/hash.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <algorithm>
#include <cmath>
#include <fmt/format.h>
#include <numeric>
#include <random>

namespace merian {

namespace {
constexpr const char* SHADER_MODULE = "merian-graph/nodes/render_restir_pt/render_restir_pt.slang";
// distinct periods decorrelate the pairs of different slots
constexpr std::array<uint32_t, RESTIR_PT_MAX_NEIGHBORS> PAIRING_PERIODS = {254, 238, 218, 200, 184};

uint32_t pack_delta(const int dx, const int dy) {
    return static_cast<uint32_t>(dx + 128) | (static_cast<uint32_t>(dy + 128) << 8);
}

// A perfect matching on a periodic tile whose offsets follow a normal distribution with mean
// distance `radius`: every pixel holds the offset to its partner, which holds the opposite one.
std::vector<uint32_t> generate_pairing(const float radius, const uint32_t seed) {
    std::vector<uint32_t> tables((RESTIR_PT_PAIRING_EXTENT * RESTIR_PT_PAIRING_EXTENT + 1) *
                                     RESTIR_PT_MAX_NEIGHBORS,
                                 pack_delta(0, 0));
    // the mean of a Rayleigh distribution is sigma sqrt(pi / 2)
    const float sigma = std::max(radius, 1.f) / std::sqrt(0.5f * static_cast<float>(M_PI));
    constexpr int max_delta = 127;

    for (uint32_t slot = 0; slot < RESTIR_PT_MAX_NEIGHBORS; slot++) {
        const int period = static_cast<int>(PAIRING_PERIODS[slot]);
        const int count = period * period;
        std::mt19937 rng(hash_val(seed, slot));
        std::normal_distribution<float> normal(0.f, sigma);

        std::vector<int> order(count);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        std::vector<uint8_t> matched(count, 0);
        uint32_t* table = &tables[RESTIR_PT_PAIRING_EXTENT * RESTIR_PT_PAIRING_EXTENT * slot];
        const auto match = [&](const int p, const int dx, const int dy) {
            const int px = p % period;
            const int py = p / period;
            const int q =
                ((py + dy + period * 2) % period) * period + (px + dx + period * 2) % period;
            if (q == p || matched[q] != 0) {
                return false;
            }
            matched[p] = matched[q] = 1;
            table[p] = pack_delta(dx, dy);
            table[q] = pack_delta(-dx, -dy);
            return true;
        };

        // 1. draw partners from the distribution
        for (const int p : order) {
            for (int attempt = 0; attempt < 64 && matched[p] == 0; attempt++) {
                const int dx = static_cast<int>(std::lround(normal(rng)));
                const int dy = static_cast<int>(std::lround(normal(rng)));
                if ((dx != 0 || dy != 0) && std::abs(dx) <= max_delta &&
                    std::abs(dy) <= max_delta) {
                    match(p, dx, dy);
                }
            }
        }
        // 2. the few left over take the nearest free pixel
        for (const int p : order) {
            for (int r = 1; r <= max_delta && matched[p] == 0; r++) {
                for (int d = -r; d <= r && matched[p] == 0; d++) {
                    match(p, d, -r) || match(p, d, r) || match(p, -r, d) || match(p, r, d);
                }
            }
        }
        tables[RESTIR_PT_PAIRING_EXTENT * RESTIR_PT_PAIRING_EXTENT * RESTIR_PT_MAX_NEIGHBORS +
               slot] = PAIRING_PERIODS[slot];
    }
    return tables;
}

vk::BufferCreateInfo buffer_create_info(const vk::DeviceSize size) {
    return vk::BufferCreateInfo{
        {},
        size,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer};
}

} // namespace

RenderRestirPT::RenderRestirPT() = default;

DeviceSupportInfo RenderRestirPT::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_composition(GBufferLayout::complete()->get_composition());
    composition->add_module_from_path(SHADER_MODULE, true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline"}, {"rayQuery"}) &
           program.get()->query_device_support(query_info);
}

void RenderRestirPT::initialize(const ContextHandle& context,
                                const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->resource_allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    use_raygen = !context->get_device()->get_physical_device()->is_amd();
}

void RenderRestirPT::update_render_constants() {
    uint32_t mask = 0u;
    for (uint32_t bit = 0; bit < 8; ++bit) {
        if (mask_enabled[bit])
            mask |= (1u << bit);
    }

    const auto b = [](const bool value) { return value ? "true" : "false"; };
    composition->add_module_from_string(
        "render_restir_pt_constants",
        fmt::format("namespace merian {{\n"
                    "export static const int merian_restir_pt_spp = {};\n"
                    "export static const int merian_restir_pt_max_path_length = {};\n"
                    "export static const bool merian_restir_pt_russian_roulette = {};\n"
                    "export static const uint merian_restir_pt_shift_mapping = {}u;\n"
                    "export static const bool merian_restir_pt_emission_on_primary = {};\n"
                    "export static const bool merian_restir_pt_area = {};\n"
                    "export static const uint merian_restir_pt_instance_mask = {}u;\n"
                    "export static const float merian_restir_pt_min_footprint = {:e};\n"
                    "export static const float merian_restir_pt_footprint_jitter = {:f};\n"
                    "export static const float merian_restir_pt_min_roughness = {:f};\n"
                    "export static const float merian_restir_pt_roughness_jitter = {:f};\n"
                    "export static const uint merian_restir_pt_neighbor_selection = {}u;\n"
                    "export static const int merian_restir_pt_neighbor_count = {};\n"
                    "export static const int merian_restir_pt_cgns_candidates = {};\n"
                    "export static const bool merian_restir_pt_cgns_early_stopping = {};\n"
                    "export static const bool merian_restir_pt_geometry_rejection = {};\n"
                    "export static const float merian_restir_pt_reject_normal = {:f};\n"
                    "export static const float merian_restir_pt_reject_depth = {:f};\n"
                    "export static const bool merian_restir_pt_demodulate_albedo = {};\n"
                    "export static const bool merian_restir_pt_stochastic_backprojection = {};\n"
                    "export static const bool merian_restir_pt_disocclusion_motion = {};\n"
                    "export static const bool merian_restir_pt_duplication_cap = {};\n"
                    "export static const bool merian_restir_pt_decoupled_shading = {};\n"
                    "export static const bool merian_restir_pt_dynamic_scene = {};\n"
                    "export static const uint merian_restir_pt_temporal_mode = {}u;\n"
                    "export static const bool merian_restir_pt_specular_motion = {};\n"
                    "export static const float merian_restir_pt_specular_motion_roughness = {:f};\n"
                    "export static const uint merian_restir_pt_debug_view = {}u;\n"
                    "}}",
                    spp, max_path_length, b(russian_roulette), shift_mapping,
                    b(emission_on_primary), b(area), mask, min_footprint / 100.f, footprint_jitter,
                    min_roughness, roughness_jitter, neighbor_selection, neighbor_count,
                    cgns_candidates, b(early_stopping), b(geometry_rejection), reject_normal,
                    reject_depth, b(demodulate_albedo), b(stochastic_backprojection),
                    b(disocclusion_motion), b(duplication_cap), b(decoupled_shading),
                    b(dynamic_scene), temporal_mode, b(specular_motion), specular_motion_roughness,
                    debug_view));
}

std::vector<InputConnectorDescriptor> RenderRestirPT::describe_inputs() {
    const GBufferGroup surface = {{GBufferField::Normal, GBufferField::LinearZ}};
    std::vector<GBufferGroup> gbuffer_groups = {
        {{GBufferField::Hit}}, surface, {{GBufferField::MotionVectors}}};
    if (demodulate_albedo) {
        gbuffer_groups.push_back({{GBufferField::Albedo}});
    }
    if (specular_motion) {
        gbuffer_groups.push_back({{GBufferField::SpecularHitDistance, GBufferField::Roughness}});
        gbuffer_groups.push_back({{GBufferField::DiffuseAlbedo}});
        gbuffer_groups.push_back({{GBufferField::SpecularAlbedo}});
    }
    con_gbuffer = GBufferIn::create(gbuffer_groups);
    con_prev_gbuffer = GBufferIn::create({{{GBufferField::Hit}}, surface});

    return {{.name = "scene", .connector = con_scene},
            {.name = "gbuffer", .connector = con_gbuffer, .access = ConnectorAccess::compute_read},
            {.name = "prev_gbuffer",
             .connector = con_prev_gbuffer,
             .access = ConnectorAccess::compute_read,
             .delay = 1},
            {.name = "prev_reservoirs",
             .connector = con_prev_reservoirs,
             .access = ConnectorAccess::compute_read,
             .delay = 1}};
}

std::vector<OutputConnectorDescriptor>
RenderRestirPT::describe_outputs(const NodeIOLayout& io_layout) {
    extent = io_layout[con_gbuffer]->get_create_info().extent;
    con_irradiance = ManagedVkImageOut::create(irradiance_format, extent);
    con_reservoirs = ManagedVkBufferOut::create(buffer_create_info(
        vk::DeviceSize(extent.width) * extent.height * sizeof(RestirPTReservoir)));
    return {{.name = "irradiance",
             .connector = con_irradiance,
             .access = ConnectorAccess::compute_read_write},
            {.name = "reservoirs",
             .connector = con_reservoirs,
             .access = ConnectorAccess::compute_read_write}};
}

RenderRestirPT::NodeStatusFlags
RenderRestirPT::on_connected(const NodeIOLayout& io_layout,
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

    gbuffer_composition = io[con_gbuffer]->get_layout()->get_composition();

    const vk::DeviceSize pixels = vk::DeviceSize(extent.width) * extent.height;
    const vk::DeviceSize slots = pixels * RESTIR_PT_MAX_NEIGHBORS;
    pong_reservoirs =
        resource_allocator->create_buffer(buffer_create_info(pixels * sizeof(RestirPTReservoir)),
                                          MemoryMappingType::NONE, "ReSTIR PT reservoirs");
    history = resource_allocator->create_buffer(buffer_create_info(slots * sizeof(RestirPTSuffix)),
                                                MemoryMappingType::NONE, "ReSTIR PT history");
    splat_counts =
        resource_allocator->create_buffer(buffer_create_info(pixels * sizeof(uint32_t)),
                                          MemoryMappingType::NONE, "ReSTIR PT splat counts");
    shifts =
        resource_allocator->create_buffer(buffer_create_info(slots * 2 * sizeof(RestirPTShift)),
                                          MemoryMappingType::NONE, "ReSTIR PT shifts");
    neighbors = resource_allocator->create_buffer(buffer_create_info(slots * sizeof(uint32_t)),
                                                  MemoryMappingType::NONE, "ReSTIR PT neighbors");
    confidence_sums =
        resource_allocator->create_buffer(buffer_create_info(pixels * sizeof(float)),
                                          MemoryMappingType::NONE, "ReSTIR PT confidence sums");
    backprojected = resource_allocator->create_buffer(
        buffer_create_info(pixels * sizeof(RestirPTReservoir)), MemoryMappingType::NONE,
        "ReSTIR PT backprojected reservoirs");
    temporal_domains =
        resource_allocator->create_buffer(buffer_create_info(pixels * 4 * sizeof(uint32_t)),
                                          MemoryMappingType::NONE, "ReSTIR PT temporal domains");
    splat_samples =
        resource_allocator->create_buffer(buffer_create_info(slots * 2 * sizeof(uint32_t)),
                                          MemoryMappingType::NONE, "ReSTIR PT splat samples");
    queue = resource_allocator->create_buffer(
        buffer_create_info(sizeof(RestirPTQueueHeader) + slots * 2 * sizeof(uint32_t)),
        MemoryMappingType::NONE, "ReSTIR PT shift queue");
    pairing = resource_allocator->create_buffer(
        buffer_create_info((RESTIR_PT_PAIRING_EXTENT * RESTIR_PT_PAIRING_EXTENT + 1) *
                           RESTIR_PT_MAX_NEIGHBORS * sizeof(uint32_t)),
        MemoryMappingType::NONE, "ReSTIR PT pairing");
    for (BufferHandle& d : duplication) {
        d = resource_allocator->create_buffer(buffer_create_info(pixels * sizeof(uint32_t)),
                                              MemoryMappingType::NONE, "ReSTIR PT duplication");
        submission.get_cmd()->fill(d);
    }
    pairing_dirty = true;

    // the first temporal pass resamples from reservoirs that no pass has written yet
    submission.get_cmd()->fill(io[con_reservoirs]);
    submission.get_cmd()->fill(pong_reservoirs);

    if (const SceneHandle& scene = io[con_scene]; scene && scene->is_ready()) {
        ensure_pipeline(scene);
    }

    return {};
}

void RenderRestirPT::upload_pairing(Submission& submission) {
    const std::vector<uint32_t> tables = generate_pairing(spatial_radius, seed);
    resource_allocator->get_staging()->cmd_to_device(submission.get_cmd(), pairing, tables);
    submission.get_cmd()->barrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eComputeShader,
                                  pairing->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                          vk::AccessFlagBits::eShaderRead));
    pairing_dirty = false;
}

void RenderRestirPT::ensure_pipeline(const SceneHandle& scene) {
    if (composition) {
        return;
    }

    composition = SlangComposition::create();
    composition->add_composition(scene->get_composition());
    composition->add_composition(gbuffer_composition);
    composition->add_module_from_path(SHADER_MODULE, true);
    update_render_constants();
    program = SlangProgram::create(compile_context, composition);

    for (uint32_t p = 0; p < PassCount; p++) {
        const bool raygen = use_raygen && p == Initial;
        const char* name = raygen         ? "initial_rt"
                           : p == Initial ? "initial"
                           : p == Shift   ? "shift"
                                          : "resample";
        entry_points[p] = SlangProgramEntryPoint::create(program, name);

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
                if (p == Initial || p == Shift) {
                    return ComputePipeline::create(ep->get_pipeline_layout(context),
                                                   ep->specialize());
                }
                SpecializationInfoBuilder spec;
                spec.add_entry_id(0, p);
                return ComputePipeline::create(ep->get_pipeline_layout(context),
                                               ep->specialize(spec.build()));
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

[[nodiscard]] RenderRestirPT::NodeStatusFlags
RenderRestirPT::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const auto& cmd = submission.get_cmd();
    const auto& scene = io[con_scene];
    if (!scene || !scene->is_ready())
        return {};

    if (max_path_length != emitted_max_path_length) {
        emitted_max_path_length = max_path_length;
        io.send_event("bounces_changed");
    }

    ensure_pipeline(scene);
    if (pairing_dirty) {
        upload_pairing(submission);
    }
    const ShaderObjectAllocatorHandle& obj_allocator = info.get_shader_object_allocator();

    const uint64_t iteration = info.get_iteration();
    const bool temporal = temporal_enable && iteration > 0;
    const int32_t rounds = neighbor_count > 0 ? spatial_rounds : 0;
    const BufferHandle out = io[con_reservoirs];
    const BufferHandle& dup_now = duplication[iteration % 2];
    const BufferHandle& dup_prev = duplication[(iteration + 1) % 2];

    // Every resampling pass writes a new set; they alternate between the output and the scratch
    // buffer so that the last one lands on the output.
    const int32_t writes = 1 + (temporal ? 1 : 0) + rounds;
    int32_t write = 0;
    const auto next_target = [&]() -> BufferHandle {
        write++;
        return (writes - write) % 2 == 0 ? out : pong_reservoirs;
    };

    RestirPTPushConstant pc{};
    pc.reservoirs_prev = io[con_prev_reservoirs]->get_device_address();
    pc.history = history->get_device_address();
    pc.shifts = shifts->get_device_address();
    pc.neighbors = neighbors->get_device_address();
    pc.splat_counts = splat_counts->get_device_address();
    pc.confidence_sums = confidence_sums->get_device_address();
    pc.pairing = pairing->get_device_address();
    pc.duplication = dup_now->get_device_address();
    pc.prev_duplication = dup_prev->get_device_address();
    pc.backprojected = backprojected->get_device_address();
    pc.temporal_domains = temporal_domains->get_device_address();
    pc.splat_samples = splat_samples->get_device_address();
    pc.frame = static_cast<uint32_t>(iteration);
    pc.seed = seed;
    pc.spatial_radius = spatial_radius;
    pc.scale_solid_angle = scale_solid_angle;
    pc.normal_beta = normal_beta;
    pc.early_stop_cutoff = early_stop_cutoff;
    pc.history_cap = history_cap;
    pc.duplication_exponent = duplication_exponent;

    const vk::PipelineStageFlags stages =
        vk::PipelineStageFlagBits::eComputeShader |
        (use_raygen ? vk::PipelineStageFlagBits::eRayTracingShaderKHR : vk::PipelineStageFlags{});
    const auto sync = [&](const vk::PipelineStageFlags dst_stages = {}) {
        cmd->barrier(
            stages, stages | dst_stages,
            vk::MemoryBarrier{vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                  vk::AccessFlagBits::eIndirectCommandRead});
    };

    const auto run = [&](const Pass p, const BufferHandle& in, const BufferHandle& target,
                         const uint32_t flags) {
        const auto ep = entry_points[p].get();
        const auto pipe = pipelines[p].get();
        const auto obj = params[p].get();

        auto cursor = obj->get_cursor();
        cursor["gbuffer"] = io[con_gbuffer].r();
        cursor["prev_gbuffer"] = io[con_prev_gbuffer].r();
        cursor["irradiance"] = io[con_irradiance].get_texture();
        cursor["queue"] = queue;

        pc.flags = flags;
        pc.reservoirs_in = in ? in->get_device_address() : 0;
        pc.reservoirs_out = target ? target->get_device_address() : 0;

        cmd->bind(pipe);
        ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
        ep->bind("params", obj, cmd, pipe, obj_allocator);
        cmd->push_constant(pipe, pc);
        if (p == Shift) {
            cmd->dispatch_indirect(queue);
        } else if (p == Duplication) {
            cmd->dispatch(extent, 16, 16);
        } else if (use_raygen && p == Initial) {
            cmd->trace_rays(initial_sbt.get(), extent);
        } else {
            cmd->dispatch(extent, 8, 8);
        }
    };

    // neighbor selection queues the shifts, one indirect dispatch runs them
    const auto run_shifts = [&](const Pass neighbor_pass, const BufferHandle& in,
                                const uint32_t flags) {
        const RestirPTQueueHeader empty{0, 1, 1, 0};
        cmd->update(queue, 0, sizeof(empty), &empty);
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eComputeShader,
                     queue->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                           vk::AccessFlagBits::eShaderRead |
                                               vk::AccessFlagBits::eShaderWrite));
        run(neighbor_pass, in, nullptr, flags);
        sync(vk::PipelineStageFlagBits::eDrawIndirect);
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "shift");
        run(Shift, in, nullptr, flags);
        sync();
    };

    BufferHandle current = next_target();
    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "initial candidates");
        run(Initial, current == out ? pong_reservoirs : out, current,
            writes == 1 ? RestirPTPassResolves : 0u);
    }
    sync();

    if (temporal) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "temporal");
        const bool subpixel_backprojection =
            area && (temporal_mode == RestirPTTemporalSubpixelAsIs ||
                     temporal_mode == RestirPTTemporalSubpixelCoverage);
        if (subpixel_backprojection) {
            if (temporal_mode == RestirPTTemporalSubpixelCoverage) {
                run_shifts(CoverageNeighbors, current, RestirPTPassCoverageShifts);
            }
            run(SubpixelBackprojection, current, nullptr, 0u);
            sync();
        }
        if (temporal_mode == RestirPTTemporalSplatting ||
            temporal_mode == RestirPTTemporalSplattingFallback) {
            cmd->fill(splat_counts);
            cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                         vk::PipelineStageFlagBits::eComputeShader,
                         splat_counts->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                      vk::AccessFlagBits::eShaderRead |
                                                          vk::AccessFlagBits::eShaderWrite));
            run(Splat, current, nullptr, 0u);
            sync();
        }
        run_shifts(TemporalNeighbors, current, RestirPTPassTemporalShifts);
        const BufferHandle target = next_target();
        run(Temporal, current, target, write == writes ? RestirPTPassResolves : 0u);
        current = target;
        sync();
    }

    for (int32_t round = 0; round < rounds; round++) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "spatial");
        pc.round = static_cast<uint32_t>(round);
        pc.pairing_transform = static_cast<uint32_t>(hash_val(iteration, round, seed));
        run_shifts(SpatialNeighbors, current, 0u);
        const BufferHandle target = next_target();
        run(Spatial, current, target, write == writes ? RestirPTPassResolves : 0u);
        current = target;
        sync();
    }

    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "duplication");
        run(Duplication, current, nullptr, 0u);
    }

    return {};
}

RenderRestirPT::NodeStatusFlags RenderRestirPT::properties(Properties& config) {
    bool needs_reconnect = false;
    bool constants_changed = false;

    constants_changed |=
        config.config_int("samples per pixel", spp, "Paths traced per pixel.", 1, 16);
    if (config.config_uint("seed", seed, "Base seed of every random number.")) {
        pairing_dirty = true;
    }
    constants_changed |=
        config.config_int("max path length", max_path_length,
                          "Maximum number of path segments, including the primary hit.", 2, 16);
    constants_changed |= config.config_bool(
        "russian roulette", russian_roulette,
        "Terminate low-throughput paths. The target function leaves the roulette out, so it only "
        "decides which candidates are drawn.");
    constants_changed |=
        config.config_bool("emission on primary", emission_on_primary,
                           "Fold primary-hit emission into irradiance (self-contained). "
                           "Otherwise it is the GBuffer emission texture's job.");
    constants_changed |= config.config_bool(
        "area reservoirs", area,
        "Integrate over the pixel's area and the lens (Zhang et al. 2024): every path carries its "
        "own point of the pixel, and a shift traces the primary ray through the same point of the "
        "destination pixel.");

    if (config.st_begin_child("shift", "Shift mapping")) {
        constants_changed |= config.config_options(
            "mapping", shift_mapping, {"reconnection", "hybrid"}, Properties::OptionsStyle::COMBO,
            "'reconnection' reconnects at the second vertex. 'hybrid' replays the random numbers "
            "up to the first vertex whose lobe before it is rough and whose segment is long "
            "compared to the pixel footprint, and reconnects there (Lin et al. 2026, 4).");
        if (shift_mapping == RestirPTShiftHybrid) {
            constants_changed |= config.config_float(
                "min footprint", min_footprint,
                "Footprint a segment needs to be reconnected, in percent of the pixel footprint "
                "(Lin et al. 2026, eq. 5). Shorter segments are replayed.",
                0.f);
            constants_changed |= config.config_float(
                "footprint jitter", footprint_jitter,
                "Relative jitter of the minimum footprint per path, which hides the boundary "
                "between replay and reconnection.",
                0.f, 1.f);
            constants_changed |= config.config_float(
                "min roughness", min_roughness,
                "Linear roughness the lobe before a vertex needs for the vertex to be reconnected.",
                0.f, 1.f);
            constants_changed |=
                config.config_float("roughness jitter", roughness_jitter,
                                    "Relative jitter of the minimum roughness per path.", 0.f, 1.f);
        }
        config.st_end_child();
    }

    if (config.st_begin_child("temporal", "Temporal reuse")) {
        config.config_bool("enable", temporal_enable);
        constants_changed |= config.config_options(
            "reprojection", temporal_mode,
            {"backprojection", "subpixel backprojection, samples as-is",
             "subpixel backprojection, guaranteed coverage", "splatting",
             "splatting, backprojection fallback"},
            Properties::OptionsStyle::COMBO,
            "How the previous frame's reservoirs reach a pixel. 'backprojection' takes the "
            "reservoir of the pixel the motion vector points at. 'subpixel backprojection' (area "
            "reservoirs) moves the pixel's footprint back by its motion and resamples the previous "
            "samples in it (Zhang et al. 2024, 4.3): 'samples as-is' those that fall inside, "
            "'guaranteed coverage' also those of the overlapped pixels by one-pixel shifts, at "
            "about twice the cost; the latter diverges where geometry moves rigidly, since the "
            "previous frame is traced in the current scene. 'splatting' pushes every reservoir "
            "onto the pixel its primary vertex projects to (Liu et al. 2025); its fallback "
            "backprojects where nothing landed, except with area reservoirs.");
        config.config_float("history cap", history_cap,
                            "Samples a temporal reservoir may stand for. Longer histories "
                            "converge faster but correlate neighbouring pixels.",
                            1.f);
        constants_changed |= config.config_bool(
            "stochastic backprojection", stochastic_backprojection,
            "Backproject to one of the four pixels around the motion vector's end, in proportion "
            "to their overlap, instead of the nearest.");
        needs_reconnect |= config.config_bool(
            "specular motion vectors", specular_motion,
            "Backproject the specular share of a smooth surface by the motion of the virtual "
            "image its specular chain ends in, rather than by the surface's own.");
        if (specular_motion) {
            constants_changed |= config.config_float(
                "specular max roughness", specular_motion_roughness,
                "Linear roughness up to which a surface counts as smooth.", 0.f, 1.f);
        }
        constants_changed |= config.config_bool(
            "disocclusion motion vectors", disocclusion_motion,
            "Where the backprojected pixel lies behind an occluder, follow the occluder's motion "
            "instead (Zeng et al. 2021).");
        constants_changed |= config.config_bool(
            "dynamic scene update", dynamic_scene,
            "Re-trace what arrives at a history path's reconnection vertex in the current scene, "
            "and evaluate a path moved into the previous frame on that frame's geometry, so "
            "lights and geometry that moved are seen where they are in each frame.");
        constants_changed |= config.config_bool(
            "duplication cap", duplication_cap,
            "Lower the history cap where the previous frame's path repeats around the pixel (Lin "
            "et al. 2026, 6).");
        if (duplication_cap) {
            config.config_float("duplication exponent", duplication_exponent,
                                "Exponent on the duplication rate that lowers the cap.", 0.f, 10.f);
        }
        config.st_end_child();
    }

    if (config.st_begin_child("spatial", "Spatial reuse")) {
        config.config_int("rounds", spatial_rounds, "Spatial resampling passes.", 0, 8);
        constants_changed |= config.config_options(
            "neighbor selection", neighbor_selection, {"reciprocal", "disk", "CGNS"},
            Properties::OptionsStyle::COMBO,
            "'reciprocal' pairs pixels so that each shift serves both sides (Lin et al. 2026, "
            "5.1). 'disk' draws neighbors from a low-discrepancy disk. 'CGNS' selects them by how "
            "compatible their primary hits are (Junkins et al. 2026).");
        constants_changed |= config.config_int("neighbor count", neighbor_count,
                                               "Neighbors resampled per pixel and round.", 1,
                                               RESTIR_PT_MAX_NEIGHBORS);
        if (config.config_float("radius", spatial_radius,
                                "Pixel radius of the disk the neighbors are drawn from; the mean "
                                "distance of a pair with reciprocal selection.",
                                1.f, 100.f)) {
            pairing_dirty = true;
        }
        constants_changed |= config.config_bool(
            "geometry rejection", geometry_rejection,
            "Drop neighbors whose normal or depth diverges too far from the pixel's.");
        if (geometry_rejection) {
            float angle = std::acos(reject_normal);
            constants_changed |= config.config_angle(
                "max normal angle", angle, "Reject neighbors with normals farther apart.", 0, 180);
            reject_normal = std::cos(angle);
            constants_changed |= config.config_percent(
                "max depth difference", reject_depth,
                "Reject neighbors whose camera distance differs by more than this fraction.");
        }
        if (neighbor_selection == RestirPTNeighborsCGNS && config.st_begin_child("cgns", "CGNS")) {
            constants_changed |= config.config_int(
                "candidates", cgns_candidates,
                "K: candidates scored per pixel before the neighbors are kept.", 1, 128);
            config.config_float("solid angle", scale_solid_angle,
                                "Solid angle of the characteristic search disk, in steradian. "
                                "Sets the world-space falloff of the position term.",
                                0.f);
            config.config_float("normal exponent", normal_beta,
                                "Exponent on the normal compatibility term.", 0.f);
            constants_changed |= config.config_bool(
                "early stopping", early_stopping,
                "Stop scoring candidates once enough of them are compatible enough.");
            if (early_stopping) {
                config.config_float("early stop score", early_stop_cutoff,
                                    "Compatibility score a candidate must reach to count towards "
                                    "stopping the search.",
                                    0.f, 1.f);
            }
            config.st_end_child();
        }
        config.st_end_child();
    }

    if (config.st_begin_child("instance_mask", "Instance mask")) {
        for (uint32_t bit = 0; bit < 8; ++bit) {
            constants_changed |= config.config_bool(std::to_string(bit), mask_enabled[bit]);
            if ((bit & 3u) != 3u)
                config.st_no_space();
        }
        config.st_end_child();
    }

    if (config.st_begin_child("output", "Output")) {
        constants_changed |= config.config_bool(
            "demodulate albedo", demodulate_albedo,
            "Divide the primary-hit albedo out of the output so a denoiser can re-modulate after "
            "filtering.");
        constants_changed |= config.config_bool(
            "decoupled shading", decoupled_shading,
            "Shade with every resampled path by its resampling weight rather than the selected "
            "one.");
        constants_changed |= config.config_options(
            "debug view", debug_view,
            {"none", "duplication", "reconnection vertex", "confidence", "reuse success"},
            Properties::OptionsStyle::COMBO,
            "'duplication' shows the share of the window around a pixel that holds its path (red) "
            "and how long it survived (green). 'reconnection vertex' shows it over the path "
            "length (red), paths replayed whole (green) and the path length (blue). 'reuse "
            "success' shows the share of neighbors accepted (red), of their shifts that succeeded "
            "(green), and whether a neighbor's path won (blue).");
        needs_reconnect |= config.config_enum("irradiance format", irradiance_format,
                                              Properties::OptionsStyle::COMBO);
        needs_reconnect |= config.config_bool(
            "ray tracing pipeline", use_raygen,
            "Trace the candidates from a raygen shader instead of a compute shader.");
        config.st_end_child();
    }

    if (constants_changed && composition) {
        update_render_constants();
    }

    if (needs_reconnect)
        return NEEDS_RECONNECT;
    return {};
}

} // namespace merian
