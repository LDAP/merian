#include "merian-graph/nodes/render_restir_pt/render_restir_pt.hpp"

#include "merian/shader/shader_compile_context.hpp"
#include "merian/utils/hash.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_ray_tracing_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fmt/format.h>
#include <numbers>
#include <numeric>
#include <random>
#include <vector>

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
    const float sigma = std::max(radius, 1.f) / std::sqrt(0.5f * std::numbers::pi_v<float>);
    constexpr int max_delta = 127;

    for (uint32_t slot = 0; slot < RESTIR_PT_MAX_NEIGHBORS; slot++) {
        const int period = static_cast<int>(PAIRING_PERIODS[slot]);
        const int count = period * period;
        std::mt19937 rng(xxhash32(seed, slot));
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

bool is_shift(const uint32_t pass) {
    return pass == RenderRestirPT::ShiftSpatial || pass == RenderRestirPT::ShiftTemporal ||
           pass == RenderRestirPT::ShiftTemporalToPrev || pass == RenderRestirPT::ShiftCoverage;
}

uint32_t shift_kind(const uint32_t pass) {
    switch (pass) {
    case RenderRestirPT::ShiftTemporal:
        return RestirPTShiftTemporal;
    case RenderRestirPT::ShiftTemporalToPrev:
        return RestirPTShiftTemporalToPrev;
    case RenderRestirPT::ShiftCoverage:
        return RestirPTShiftCoverage;
    default:
        return RestirPTShiftSpatial;
    }
}

} // namespace

RenderRestirPT::RenderRestirPT() = default;

DeviceSupportInfo RenderRestirPT::query_device_support(const DeviceSupportQueryInfo& query_info) {
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_composition(GBufferLayout::complete()->get_composition());
    path_records.add_constants(composition);
    composition->add_module_from_path(SHADER_MODULE, true);
    const auto program = SlangProgram::create(query_info.compile_context, composition);
    return DeviceSupportInfo::check(query_info, {"rayTracingPipeline"}, {"rayQuery"}) &
           program.get()->query_device_support(query_info);
}

void RenderRestirPT::initialize(const ContextHandle& context,
                                [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
    this->compile_context = context->get_shader_compile_context();

    // the ray tracing pipeline caps the registers on AMD
    raygen_preferred = !context->get_device()->get_physical_device()->is_amd();
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
                    "export static const uint merian_restir_pt_roulette_depth = {}u;\n"
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
                    "export static const bool merian_restir_pt_independent_feedback = {};\n"
                    "export static const bool merian_restir_pt_retrace_suffix = {};\n"
                    "export static const uint merian_restir_pt_temporal_mode = {}u;\n"
                    "export static const bool merian_restir_pt_specular_motion = {};\n"
                    "export static const float merian_restir_pt_specular_motion_roughness = {:f};\n"
                    "export static const uint merian_restir_pt_debug_view = {}u;\n"
                    "export static const uint merian_restir_pt_record = {}u;\n"
                    "}}",
                    spp, max_path_length, b(russian_roulette), roulette_depth, shift_mapping,
                    b(emission_on_primary), b(area), mask, min_footprint / 100.f, footprint_jitter,
                    min_roughness, roughness_jitter, neighbor_selection, neighbor_count,
                    cgns_candidates, b(early_stopping),
                    b(geometry_rejection && neighbor_selection != RestirPTNeighborsCGNS),
                    reject_normal, reject_depth, b(demodulate_albedo), b(stochastic_backprojection),
                    b(disocclusion_motion), b(duplication_cap), b(decoupled_shading),
                    b(independent_feedback), b(retrace_suffix), temporal_mode, b(specular_motion),
                    specular_motion_roughness, debug_view, record_mode));
    path_records.add_constants(composition);
}

uint32_t RenderRestirPT::recorded_paths_per_pixel() const {
    return record_mode == RestirPTRecordShaded ? 1 : static_cast<uint32_t>(std::max(spp, 1));
}

bool RenderRestirPT::use_raygen() const {
    return trace_shader == TraceShader::Auto ? raygen_preferred
                                             : trace_shader == TraceShader::RayGeneration;
}

uint32_t RenderRestirPT::recorded_vertices_per_path() const {
    const auto length = static_cast<uint32_t>(std::max(max_path_length, 1));
    return record_mode == RestirPTRecordShaded ? length + 2 : (2 * length) + 1;
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

    // the initial candidates are traced in a ray tracing pipeline on some devices
    return {{.name = "scene", .connector = con_scene},
            {.name = "gbuffer",
             .connector = con_gbuffer,
             .access = ConnectorAccess::compute_read | ConnectorAccess::ray_tracing_read},
            {.name = "prev_gbuffer",
             .connector = con_prev_gbuffer,
             .access = ConnectorAccess::compute_read | ConnectorAccess::ray_tracing_read,
             .delay = 1,
             .auto_connect = AutoConnect::input("gbuffer")},
            {.name = "prev_reservoirs",
             .connector = con_prev_reservoirs,
             .access = ConnectorAccess::compute_read | ConnectorAccess::ray_tracing_read,
             .delay = 1,
             .auto_connect = AutoConnect::output("reservoirs")}};
}

std::vector<OutputConnectorDescriptor>
RenderRestirPT::describe_outputs(const NodeIOLayout& io_layout) {
    extent = io_layout[con_gbuffer]->get_create_info().extent;
    con_irradiance = ManagedVkImageOut::create(irradiance_format, extent);
    con_reservoirs = ManagedVkBufferOut::create(buffer_create_info(
        vk::DeviceSize(extent.width) * extent.height * sizeof(RestirPTReservoir)));
    return {
        {.name = "irradiance",
         .connector = con_irradiance,
         .access = ConnectorAccess::compute_read_write | ConnectorAccess::ray_tracing_read_write |
                   ConnectorAccess::transfer_dst},
        {.name = "reservoirs",
         .connector = con_reservoirs,
         .access = ConnectorAccess::compute_read_write | ConnectorAccess::ray_tracing_read_write},
        path_records.describe_output(io_layout.get_context(), extent, recorded_paths_per_pixel(),
                                     recorded_vertices_per_path())};
}

RenderRestirPT::NodeStatusFlags
RenderRestirPT::on_connected(const NodeIOLayout& io_layout,
                             const NodeIO& io,
                             [[maybe_unused]] const NodeConnectionInfo& info,
                             Submission& submission) {
    composition = nullptr;
    previous_iteration_traced = false;
    seed = io.get_seed();
    if (path_records.update_connected(io_layout)) {
        return NEEDS_RECONNECT;
    }

    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            if (composition) {
                if (std::any_cast<bool>(force)) {
                    composition->force_reload();
                } else {
                    composition->reload(compile_context->get_search_path_file_loader());
                }
            }
        });

    gbuffer_composition = io[con_gbuffer]->get_layout()->get_composition();

    const vk::DeviceSize pixels = vk::DeviceSize(extent.width) * extent.height;
    const vk::DeviceSize slots = pixels * RESTIR_PT_MAX_NEIGHBORS;
    const ResourceAllocatorHandle& allocator = io.get_allocator();
    pong_reservoirs =
        allocator->create_buffer(buffer_create_info(pixels * sizeof(RestirPTReservoir)),
                                 MemoryMappingType::NONE, "ReSTIR PT reservoirs");
    history = allocator->create_buffer(buffer_create_info(slots * sizeof(RestirPTSuffix)),
                                       MemoryMappingType::NONE, "ReSTIR PT history");
    splat_counts = allocator->create_buffer(buffer_create_info(pixels * sizeof(uint32_t)),
                                            MemoryMappingType::NONE, "ReSTIR PT splat counts");
    shifts = allocator->create_buffer(buffer_create_info(slots * 2 * sizeof(RestirPTShift)),
                                      MemoryMappingType::NONE, "ReSTIR PT shifts");
    neighbors = allocator->create_buffer(buffer_create_info(slots * sizeof(uint32_t)),
                                         MemoryMappingType::NONE, "ReSTIR PT neighbors");
    confidence_sums =
        allocator->create_buffer(buffer_create_info(pixels * sizeof(float)),
                                 MemoryMappingType::NONE, "ReSTIR PT confidence sums");
    backprojected =
        allocator->create_buffer(buffer_create_info(pixels * sizeof(RestirPTReservoir)),
                                 MemoryMappingType::NONE, "ReSTIR PT backprojected reservoirs");
    temporal_domains =
        allocator->create_buffer(buffer_create_info(pixels * 4 * sizeof(uint32_t)),
                                 MemoryMappingType::NONE, "ReSTIR PT temporal domains");
    splat_samples = allocator->create_buffer(buffer_create_info(pixels * 2 * sizeof(uint32_t)),
                                             MemoryMappingType::NONE, "ReSTIR PT splat samples");
    queue = allocator->create_buffer(
        buffer_create_info(RESTIR_PT_QUEUES * sizeof(RestirPTQueueHeader) +
                           pixels * RESTIR_PT_MAX_SHIFTS * sizeof(uint32_t)),
        MemoryMappingType::NONE, "ReSTIR PT shift queue");
    pairing = allocator->create_buffer(
        buffer_create_info((RESTIR_PT_PAIRING_EXTENT * RESTIR_PT_PAIRING_EXTENT + 1) *
                           RESTIR_PT_MAX_NEIGHBORS * sizeof(uint32_t)),
        MemoryMappingType::NONE, "ReSTIR PT pairing");
    for (BufferHandle& d : duplication) {
        d = allocator->create_buffer(buffer_create_info(pixels * sizeof(uint32_t)),
                                     MemoryMappingType::NONE, "ReSTIR PT duplication");
        submission.get_cmd()->fill(d);
    }
    pairing_dirty = true;

    submission.get_cmd()->fill(pong_reservoirs);

    if (const SceneHandle& scene = io[con_scene]; scene && scene->is_ready()) {
        ensure_pipeline(io.get_context(), allocator, scene);
    }

    return {};
}

void RenderRestirPT::upload_pairing(const ResourceAllocatorHandle& allocator,
                                    Submission& submission) {
    const std::vector<uint32_t> tables = generate_pairing(spatial_radius, seed);
    submission.get_cmd()->barrier(vk::PipelineStageFlagBits::eComputeShader,
                                  vk::PipelineStageFlagBits::eTransfer,
                                  pairing->buffer_barrier(vk::AccessFlagBits::eShaderRead,
                                                          vk::AccessFlagBits::eTransferWrite));
    allocator->get_staging()->cmd_to_device(submission.get_cmd(), pairing, tables);
    submission.get_cmd()->barrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eComputeShader,
                                  pairing->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                          vk::AccessFlagBits::eShaderRead));
    pairing_dirty = false;
}

void RenderRestirPT::ensure_pipeline(const ContextHandle& context,
                                     const ResourceAllocatorHandle& allocator,
                                     const SceneHandle& scene) {
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
        const bool raygen = use_raygen() && p == Initial;
        const char* name = raygen         ? "initial_rt"
                           : p == Initial ? "initial"
                           : is_shift(p)  ? "shift"
                           : p == Train   ? "train"
                                          : "resample";
        entry_points[p] = SlangProgramEntryPoint::create(program, name);

        if (raygen) {
            pipelines[p] = Versioned<Pipeline>([this, context, p] {
                const auto ep = entry_points[p].get();
                return RayTracingPipelineBuilder()
                    .add_raygen_group(ep->specialize())
                    .build(ep->get_pipeline_layout(context));
            });
            pipelines[p].depends_on(entry_points[p]);

            initial_sbt = Versioned<ShaderBindingTable>([this, allocator] {
                return ShaderBindingTable::create(
                    std::dynamic_pointer_cast<RayTracingPipeline>(pipelines[Initial].get()),
                    allocator);
            });
            initial_sbt.depends_on(pipelines[p]);
        } else {
            pipelines[p] = Versioned<Pipeline>([this, context, p] {
                const auto ep = entry_points[p].get();
                if (p == Initial || p == Train) {
                    return ComputePipeline::create(ep->get_pipeline_layout(context),
                                                   ep->specialize());
                }
                SpecializationInfoBuilder spec;
                if (is_shift(p))
                    spec.add_entry_id(1, shift_kind(p));
                else
                    spec.add_entry_id(0, p);
                return ComputePipeline::create(ep->get_pipeline_layout(context),
                                               ep->specialize(spec.build()));
            });
            pipelines[p].depends_on(entry_points[p]);
        }

        params[p] = Versioned<ShaderObject>([this, context, allocator, p] {
            return entry_points[p]->create_shader_object_for_parameter(context, "params",
                                                                       allocator);
        });
        params[p].depends_on(entry_points[p]);
    }
}

[[nodiscard]] RenderRestirPT::NodeStatusFlags
RenderRestirPT::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const auto& cmd = submission.get_cmd();
    const auto& scene = io[con_scene];
    if (!scene || !scene->is_ready()) {
        if (io.is_connected(con_irradiance)) {
            cmd->clear(io[con_irradiance]);
        }
        previous_iteration_traced = false;
        return {};
    }

    if (max_path_length != emitted_max_path_length) {
        emitted_max_path_length = max_path_length;
        io.send_event("bounces_changed");
    }

    ensure_pipeline(io.get_context(), io.get_allocator(), scene);
    path_records.process(cmd, io, info.get_iteration(), {});
    if (pairing_dirty) {
        upload_pairing(io.get_allocator(), submission);
    }
    const ShaderObjectAllocatorHandle& obj_allocator = info.get_shader_object_allocator();

    const uint64_t iteration = info.get_iteration();
    const bool previous_reservoirs = previous_iteration_traced;
    previous_iteration_traced = true;
    const bool temporal = temporal_enable && previous_reservoirs;
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
        (use_raygen() ? vk::PipelineStageFlagBits::eRayTracingShaderKHR : vk::PipelineStageFlags{});
    const auto sync = [&](const vk::PipelineStageFlags dst_stages = {}) {
        const vk::AccessFlags indirect = (dst_stages & vk::PipelineStageFlagBits::eDrawIndirect)
                                             ? vk::AccessFlagBits::eIndirectCommandRead
                                             : vk::AccessFlags{};
        cmd->barrier(
            stages, stages | dst_stages,
            vk::MemoryBarrier{vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                                  indirect});
    };

    // a pass runs more than once per frame, so its set must not change once bound
    for (uint32_t p = 0; p < PassCount; p++) {
        auto cursor = params[p]->get_cursor();
        cursor["gbuffer"] = io[con_gbuffer].r();
        cursor["prev_gbuffer"] = io[con_prev_gbuffer].r();
        cursor["irradiance"] = io[con_irradiance].get_texture();
        cursor["queue"] = queue;
        path_records.bind(io, cursor.find("path_records"));
    }

    const auto run = [&](const Pass p, const BufferHandle& in, const BufferHandle& target,
                         const uint32_t flags) {
        const auto ep = entry_points[p].get();
        const auto pipe = pipelines[p].get();
        const auto obj = params[p].get();

        pc.flags = flags;
        pc.reservoirs_in = in ? in->get_device_address() : 0;
        pc.reservoirs_out = target ? target->get_device_address() : 0;

        cmd->bind(pipe);
        ep->bind("scene", scene->get_shader_object(), cmd, pipe, obj_allocator);
        ep->bind("params", obj, cmd, pipe, obj_allocator);
        cmd->push_constant(pipe, pc);
        if (is_shift(p)) {
            const uint32_t queue_index =
                p == ShiftTemporalToPrev ? RESTIR_PT_INTO_NEIGHBOR : RESTIR_PT_INTO_PIXEL;
            cmd->dispatch_indirect(queue, queue_index * sizeof(RestirPTQueueHeader));
        } else if (p == Train) {
            const uint32_t pixels = extent.width * extent.height;
            const uint32_t threads = (pixels + RESTIR_PT_TRAIN_STRIDE - 1) / RESTIR_PT_TRAIN_STRIDE;
            cmd->dispatch((threads + RESTIR_PT_TRAIN_GROUP - 1) / RESTIR_PT_TRAIN_GROUP, 1, 1);
        } else if (use_raygen() && p == Initial) {
            cmd->trace_rays(initial_sbt.get(), extent);
        } else {
            cmd->dispatch(extent, 8, 8);
        }
    };

    const auto before_transfer = [&](const BufferHandle& buffer) {
        cmd->barrier(stages | vk::PipelineStageFlagBits::eDrawIndirect,
                     vk::PipelineStageFlagBits::eTransfer,
                     buffer->buffer_barrier(vk::AccessFlagBits::eShaderRead |
                                                vk::AccessFlagBits::eShaderWrite |
                                                vk::AccessFlagBits::eIndirectCommandRead,
                                            vk::AccessFlagBits::eTransferWrite));
    };

    const auto run_shifts = [&](const Pass neighbor_pass, const BufferHandle& in,
                                const std::initializer_list<Pass> shift_passes) {
        std::array<RestirPTQueueHeader, RESTIR_PT_QUEUES> empty;
        empty.fill({0, 1, 1, 0});
        before_transfer(queue);
        cmd->update(queue, 0, sizeof(empty), empty.data());
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eComputeShader,
                     queue->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                           vk::AccessFlagBits::eShaderRead |
                                               vk::AccessFlagBits::eShaderWrite));
        run(neighbor_pass, in, nullptr, 0u);
        sync(vk::PipelineStageFlagBits::eDrawIndirect);
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "shift");
        for (const Pass shift_pass : shift_passes)
            run(shift_pass, in, nullptr, 0u);
        sync();
    };

    if (independent_feedback && scene->get_light_collection().densities_vary()) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "light training");
        run(Train, nullptr, nullptr, 0u);
    }
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
                run_shifts(CoverageNeighbors, current, {ShiftCoverage});
            }
            run(SubpixelBackprojection, current, nullptr, 0u);
            sync();
        }
        if (temporal_mode == RestirPTTemporalSplatting ||
            temporal_mode == RestirPTTemporalSplattingFallback) {
            before_transfer(splat_counts);
            before_transfer(neighbors);
            cmd->fill(splat_counts);
            cmd->fill(neighbors, RESTIR_PT_NO_NEIGHBOR);
            cmd->barrier(vk::PipelineStageFlagBits::eTransfer,
                         vk::PipelineStageFlagBits::eComputeShader,
                         vk::MemoryBarrier{vk::AccessFlagBits::eTransferWrite,
                                           vk::AccessFlagBits::eShaderRead |
                                               vk::AccessFlagBits::eShaderWrite});
            run(Splat, current, nullptr, 0u);
            sync();
        }
        run_shifts(TemporalNeighbors, current, {ShiftTemporal, ShiftTemporalToPrev});
        const BufferHandle target = next_target();
        run(Temporal, current, target, write == writes ? RestirPTPassResolves : 0u);
        current = target;
        sync();
    }

    for (int32_t round = 0; round < rounds; round++) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "spatial");
        pc.round = static_cast<uint32_t>(round);
        pc.pairing_transform =
            xxhash32(static_cast<uint32_t>(iteration), static_cast<uint32_t>(round), seed);
        run_shifts(SpatialNeighbors, current, {ShiftSpatial});
        const BufferHandle target = next_target();
        run(Spatial, current, target, write == writes ? RestirPTPassResolves : 0u);
        current = target;
        sync();
    }

    if (path_records.connected() && record_mode == RestirPTRecordShaded) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "record");
        run(Record, current, nullptr, 0u);
        sync();
    }

    if (duplication_cap || debug_view == RestirPTDebugDuplication) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "duplication");
        run(Duplication, current, nullptr,
            previous_reservoirs ? RestirPTPassPreviousReservoirs : 0u);
    }

    return {};
}

RenderRestirPT::NodeStatusFlags RenderRestirPT::properties(Properties& config) {
    bool needs_reconnect = false;
    bool constants_changed = false;

    constants_changed |=
        config.config_int("samples per pixel", spp, "Paths traced per pixel.", 1, 16);
    constants_changed |=
        config.config_int("max path length", max_path_length,
                          "Maximum number of path segments, including the primary hit.", 2, 16);
    constants_changed |= config.config_bool(
        "russian roulette", russian_roulette,
        "Terminate low-throughput paths. The target function leaves the roulette out, so it only "
        "decides which candidates are drawn.");
    if (russian_roulette || !config.is_ui()) {
        constants_changed |= config.config_uint(
            "roulette depth", roulette_depth,
            "Path depth, counted from the primary hit, up to which no path is terminated.", 0, 16);
    }
    constants_changed |= config.config_bool(
        "area reservoirs", area,
        "Integrate over the pixel's area and the lens (Zhang et al. 2024): every path carries its "
        "own point of the pixel, and a shift traces the primary ray through the same point of the "
        "destination pixel.");
    constants_changed |= config.config_bool(
        "decoupled shading", decoupled_shading,
        "Shade with every resampled path by its resampling weight rather than the selected one.");

    if (config.st_begin_child("shift", "Shift mapping")) {
        constants_changed |= config.config_options(
            "mapping", shift_mapping, {"reconnection", "hybrid"}, Properties::OptionsStyle::COMBO,
            "'reconnection' reconnects at the second vertex. 'hybrid' replays the random numbers "
            "up to the first vertex whose BSDF before it is rough and whose segment is long "
            "compared to the pixel footprint, and reconnects there (Lin et al. 2026, 4).");
        if (shift_mapping == RestirPTShiftHybrid) {
            constants_changed |= config.config_float(
                "min footprint", min_footprint,
                "Footprint a segment needs to be reconnected, in percent of the pixel footprint "
                "(Lin et al. 2026, eq. 5). Shorter segments are replayed.",
                0.1f, 0.f);
            constants_changed |= config.config_float(
                "footprint jitter", footprint_jitter,
                "Relative jitter of the minimum footprint per path, which hides the boundary "
                "between replay and reconnection.",
                0.01f, 0.f, 1.f);
            constants_changed |= config.config_float(
                "min roughness", min_roughness,
                "Linear roughness the BSDF before a vertex needs for the vertex to be reconnected.",
                0.01f, 0.f, 1.f);
            constants_changed |= config.config_float(
                "roughness jitter", roughness_jitter,
                "Relative jitter of the minimum roughness per path.", 0.01f, 0.f, 1.f);
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
            "'guaranteed coverage' also those of the overlapped pixels by one-pixel shifts (Zhang "
            "et al. 2024, 4.3.2), at about twice the cost. 'splatting' pushes every reservoir onto "
            "the pixel its primary vertex projects to (Liu et al. 2025); its fallback backprojects "
            "where nothing landed, except with area reservoirs.");
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
                                0.1f, 1.f, 100.f)) {
            pairing_dirty = true;
        }
        const bool show_rejection = neighbor_selection != RestirPTNeighborsCGNS || !config.is_ui();
        if (show_rejection)
            constants_changed |= config.config_bool(
                "geometry rejection", geometry_rejection,
                "Drop neighbors whose normal or depth diverges too far from the pixel's.");
        if (show_rejection && geometry_rejection) {
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

    if (config.st_begin_child("biasing", "Biasing")) {
        constants_changed |= config.config_bool(
            "independent light feedback", independent_feedback,
            "Train the light cut with separate paths from a tenth of the pixels instead of with "
            "the light samples of the paths the reservoirs reuse.");
        constants_changed |= config.config_bool(
            "re-trace suffixes", retrace_suffix,
            "Re-trace the part of a temporal path after its reconnection vertex in the frame it "
            "is evaluated in, instead of keeping what arrived there when it was sampled. Biased "
            "when off unless the scene, its lights and their sampling stay fixed.");
        config.st_end_child();
    }

    if (config.st_begin_child("instance mask", "Instance mask")) {
        for (uint32_t bit = 0; bit < 8; ++bit) {
            constants_changed |= config.config_bool(std::to_string(bit), mask_enabled[bit]);
            if ((bit & 3u) != 3u)
                config.st_no_space();
        }
        config.st_end_child();
    }

    if (config.st_begin_child("output", "Output")) {
        constants_changed |=
            config.config_bool("emission on primary", emission_on_primary,
                               "Fold primary-hit emission into irradiance (self-contained). "
                               "Otherwise it is the GBuffer emission texture's job.");
        constants_changed |= config.config_bool(
            "demodulate albedo", demodulate_albedo,
            "Divide the primary-hit albedo out of the output so a denoiser can re-modulate after "
            "filtering.");
        constants_changed |= config.config_options(
            "show", debug_view,
            {"radiance", "duplication", "reconnection vertex", "confidence", "reuse success"},
            Properties::OptionsStyle::COMBO,
            "What the output holds. 'duplication' shows the share of the window around a pixel "
            "that holds its path (red) "
            "and how long it survived (green). 'reconnection vertex' shows it over the path "
            "length (red), paths replayed whole (green) and the path length (blue). 'reuse "
            "success' shows the share of neighbors accepted (red), of their shifts that succeeded "
            "(green), and whether a neighbor's path won (blue).");
        needs_reconnect |= config.config_enum("irradiance format", irradiance_format,
                                              Properties::OptionsStyle::COMBO);
        config.st_end_child();
    }

    if (config.st_begin_child("other", "Other")) {
        int trace_shader_index = static_cast<int>(trace_shader);
        if (config.config_options(
                "trace shader", trace_shader_index, {"auto", "ray generation", "compute"},
                Properties::OptionsStyle::COMBO,
                "Trace the candidates from a ray generation shader or a compute shader. The "
                "compute shader avoids the ray tracing pipeline's register cap; 'auto' picks it on "
                "AMD.")) {
            trace_shader = static_cast<TraceShader>(trace_shader_index);
            needs_reconnect = true;
        }
        config.st_end_child();
    }

    needs_reconnect |=
        path_records.properties(config, recorded_paths_per_pixel(), recorded_vertices_per_path());
    if (config.config_options("debugger records", record_mode, {"candidates", "shaded paths"},
                              Properties::OptionsStyle::COMBO,
                              "'candidates' are the paths the initial pass traces, with their "
                              "densities. 'shaded paths' records the path in the reservoir each "
                              "pixel shades, as the shift into the pixel traces it.")) {
        constants_changed = true;
        needs_reconnect = true;
    }

    if (constants_changed && composition) {
        update_render_constants();
    }

    if (needs_reconnect)
        return NEEDS_RECONNECT;
    return {};
}

} // namespace merian
