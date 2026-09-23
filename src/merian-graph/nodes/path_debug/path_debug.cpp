#include "merian-graph/nodes/path_debug/path_debug.hpp"

#include "merian-shaders/debug/path_record.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_graphics_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/blits.hpp"
#include "merian/vk/utils/image_export.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <fmt/format.h>
#include <imgui.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>

namespace merian {

namespace {
constexpr const char* SELECT_MODULE = "merian-graph/nodes/path_debug/select.slang";
constexpr const char* THRESHOLD_MODULE = "merian-graph/nodes/path_debug/threshold.slang";
constexpr const char* COLLECT_MODULE = "merian-graph/nodes/path_debug/collect.slang";
constexpr const char* OVERLAY_MODULE = "merian-graph/nodes/path_debug/overlay_raster.slang";
constexpr const char* COMPOSE_MODULE = "merian-graph/nodes/path_debug/compose.slang";
constexpr const char* STATS_MODULE = "merian-graph/nodes/path_debug/stats.slang";
constexpr const char* MAP_REDUCE_MODULE = "merian-graph/nodes/path_debug/map_reduce.slang";
constexpr const char* MAP_VIEW_MODULE = "merian-graph/nodes/path_debug/map_view.slang";
constexpr const char* BSDF_MAP_MODULE = "merian-graph/nodes/path_debug/bsdf_map.slang";
constexpr const char* BSDF_CHECK_MODULE = "merian-graph/nodes/path_debug/bsdf_check.slang";
constexpr const char* HEAT_DECAY_MODULE = "merian-graph/nodes/path_debug/heat_decay.slang";
constexpr const char* HEAT_SPLAT_MODULE = "merian-graph/nodes/path_debug/heat_splat.slang";
constexpr const char* FINALIZE_MODULE = "merian-graph/nodes/path_debug/finalize.slang";
constexpr const char* FILTER_RENDER_MODULE = "merian-graph/nodes/path_debug/filter_render.slang";

constexpr std::array<const char*, 12> STANDALONE_MODULES = {
    SELECT_MODULE,     THRESHOLD_MODULE,  COLLECT_MODULE,       COMPOSE_MODULE,
    STATS_MODULE,      MAP_REDUCE_MODULE, MAP_VIEW_MODULE,      HEAT_DECAY_MODULE,
    HEAT_SPLAT_MODULE, FINALIZE_MODULE,   FILTER_RENDER_MODULE, OVERLAY_MODULE};

vk::BufferMemoryBarrier2 compute_to_transfer(const BufferHandle& buffer) {
    return buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eComputeShader |
                                       vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
                                   vk::PipelineStageFlagBits2::eTransfer,
                                   vk::AccessFlagBits2::eShaderRead |
                                       vk::AccessFlagBits2::eShaderWrite,
                                   vk::AccessFlagBits2::eTransferWrite);
}

vk::BufferMemoryBarrier2 transfer_to_compute(const BufferHandle& buffer) {
    return buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

vk::BufferMemoryBarrier2 compute_to_compute(const BufferHandle& buffer) {
    return buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eComputeShader,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

uint32_t groups(const uint32_t threads) {
    return (threads + PATH_DEBUG_WG - 1) / PATH_DEBUG_WG;
}
} // namespace

PathDebugParams PathDebugNode::default_params() {
    PathDebugParams defaults{};
    defaults.overlay_alpha = 0.85f;
    defaults.overlay_thickness = 2.f;
    defaults.overlay_color_scale = 1.f;
    defaults.keep_prob = 1.f;
    defaults.record_pixel = PATH_RECORD_ALL_PIXELS;
    defaults.map_res = 32;
    defaults.map_panel_size = 192;
    defaults.map_exposure = 1.f;
    defaults.sphere_radius = 0.05f;
    defaults.heat_bounce_max = 30;
    defaults.heat_decay = 1.f;
    defaults.heat_cell_size = 0.05f;
    defaults.heat_exposure = 1.f;
    defaults.heat_opacity = 0.6f;
    defaults.grid_slots = 1u << 20;
    defaults.pixel_stats_accumulate = 1;
    defaults.pixel_stats_scale = 1.f;
    defaults.cursor_x = PATH_DEBUG_NO_PIXEL;
    defaults.cursor_y = PATH_DEBUG_NO_PIXEL;
    defaults.sel_material = PATH_RECORD_MATERIAL_NONE;
    defaults.backdrop_split = 0.5f;
    defaults.difference_gain = 10.f;
    defaults.filtered_exposure = 1.f;
    defaults.overlay_isolate = PATH_DEBUG_NO_ISOLATE;
    return defaults;
}

DeviceSupportInfo PathDebugNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo support{true};
    support.required_features.emplace_back("vertexPipelineStoresAndAtomics");
    support = support & DeviceSupportInfo::check(query_info, {}, {"shaderBufferFloat32AtomicAdd"});
    const auto gbuffer_layout = GBufferLayout::complete()->get_composition();
    for (const char* module : STANDALONE_MODULES) {
        const auto composition = SlangComposition::create();
        composition->add_composition(gbuffer_layout);
        composition->add_module_from_path(module, true);
        support = support & SlangProgram::create(query_info.compile_context, composition)
                                .get()
                                ->query_device_support(query_info);
    }
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_composition(gbuffer_layout);
    composition->add_module_from_path(BSDF_MAP_MODULE, true);
    composition->add_module_from_path(BSDF_CHECK_MODULE, true);
    return support & SlangProgram::create(query_info.compile_context, composition)
                         .get()
                         ->query_device_support(query_info);
}

void PathDebugNode::initialize(const ContextHandle& context,
                               const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
    this->compile_context = context->get_shader_compile_context();

    spec_info.set(SpecializationInfoBuilder().build());

    select_kernel.emplace(context, allocator, compile_context, SELECT_MODULE, spec_info);
    threshold_kernel.emplace(context, allocator, compile_context, THRESHOLD_MODULE, spec_info);
    collect_kernel.emplace(context, allocator, compile_context, COLLECT_MODULE, spec_info);
    map_reduce_kernel.emplace(context, allocator, compile_context, MAP_REDUCE_MODULE, spec_info);
    map_view_kernel.emplace(context, allocator, compile_context, MAP_VIEW_MODULE, spec_info);
    heat_decay_kernel.emplace(context, allocator, compile_context, HEAT_DECAY_MODULE, spec_info);
    heat_splat_kernel.emplace(context, allocator, compile_context, HEAT_SPLAT_MODULE, spec_info);
    finalize_kernel.emplace(context, allocator, compile_context, FINALIZE_MODULE, spec_info);
    filter_render_kernel.emplace(context, allocator, compile_context, FILTER_RENDER_MODULE,
                                 spec_info);
    build_gbuffer_kernels();

    state_buffer = allocator->create_buffer(vk::DeviceSize{STATE_UINTS} * 4,
                                            vk::BufferUsageFlagBits::eStorageBuffer |
                                                vk::BufferUsageFlagBits::eTransferDst,
                                            MemoryMappingType::NONE, "path_debug state");
    resampled_buffer = allocator->create_buffer(vk::DeviceSize{PATH_DEBUG_RESAMPLED_UINTS} * 4,
                                                vk::BufferUsageFlagBits::eStorageBuffer |
                                                    vk::BufferUsageFlagBits::eTransferDst,
                                                MemoryMappingType::NONE, "path_debug resampled");
    out_state_buffer = allocator->create_buffer(vk::DeviceSize{OUT_UINTS} * 4,
                                                vk::BufferUsageFlagBits::eStorageBuffer |
                                                    vk::BufferUsageFlagBits::eTransferSrc,
                                                MemoryMappingType::NONE, "path_debug out state");
    stats_buffer = allocator->create_buffer(vk::DeviceSize{STATS_UINTS} * 4,
                                            vk::BufferUsageFlagBits::eStorageBuffer |
                                                vk::BufferUsageFlagBits::eTransferDst |
                                                vk::BufferUsageFlagBits::eTransferSrc,
                                            MemoryMappingType::NONE, "path_debug stats");
    ::ImGuiContext* const prev_imgui_ctx = ImGui::GetCurrentContext();
    imgui_ctx = std::make_shared<ImGuiContext>();
    imgui_renderer = std::make_shared<ImGuiRenderer>(context, allocator, imgui_ctx);
    imgui_backend = std::make_shared<ImGuiMerianBackend>(imgui_ctx);
    ImGuiContext::set_current(prev_imgui_ctx);
}

void PathDebugNode::build_gbuffer_kernels() {
    const bool connected = gbuffer_connected;
    const auto layout = gbuffer_composition;
    const auto make = [connected, layout](const char* module) {
        const auto composition = SlangComposition::create();
        composition->add_composition(layout);
        composition->add_module_from_path(module, true);
        composition->add_module_from_string(
            "path_debug_constants",
            fmt::format("export static const bool merian_path_debug_gbuffer = {};",
                        connected ? "true" : "false"));
        return composition;
    };
    stats_kernel.emplace(
        context, allocator, compile_context, [make]() { return make(STATS_MODULE); }, spec_info);
    compose_kernel.emplace(
        context, allocator, compile_context, [make]() { return make(COMPOSE_MODULE); }, spec_info);

    overlay_composition = make(OVERLAY_MODULE);
    overlay_program = SlangProgram::create(compile_context, overlay_composition);
    overlay_vertex = SlangProgramEntryPoint::create(overlay_program, "vertex_main");
    overlay_fragment = SlangProgramEntryPoint::create(overlay_program, "fragment_main");
    overlay_globals = Versioned<ShaderObject>(
        [this] { return overlay_vertex.get()->create_global_shader_object(context, allocator); });
    overlay_globals.depends_on(overlay_vertex);
    overlay_format = vk::Format::eUndefined;
}

void PathDebugNode::build_overlay_pipeline(const vk::Format color_format) {
    overlay_format = color_format;
    overlay_pipeline = Versioned<Pipeline>([this, color_format] {
        const auto vertex = overlay_vertex.get();
        return GraphicsPipelineBuilder()
            .set_vertex_shader(vertex->specialize())
            .set_fragment_shader(overlay_fragment.get()->specialize())
            .input_assembly_topology(vk::PrimitiveTopology::eTriangleList)
            .rasterizer_cull_mode(vk::CullModeFlagBits::eNone)
            .blend_add_attachment(VK_TRUE, vk::BlendFactor::eSrcAlpha,
                                  vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
                                  vk::BlendFactor::eOne, vk::BlendFactor::eOneMinusSrcAlpha,
                                  vk::BlendOp::eAdd)
            .dyanmic_state_add(vk::DynamicState::eViewport)
            .dyanmic_state_add(vk::DynamicState::eScissor)
            .viewport_add(1.f, 1.f) // count only, the dynamic state supplies the extent
            .build_dynamic_rendering(vertex->get_pipeline_layout(context), color_format);
    });
    overlay_pipeline.depends_on(overlay_vertex);
    overlay_pipeline.depends_on(overlay_fragment);
}

void PathDebugNode::ensure_scene_pipeline(ScenePipeline& kernel,
                                          const SceneHandle& scene,
                                          const char* module) {
    if (kernel.composition) {
        return;
    }
    kernel.composition = SlangComposition::create();
    kernel.composition->add_composition(scene->get_composition());
    kernel.composition->add_composition(gbuffer_composition);
    kernel.composition->add_module_from_path(module, true);

    kernel.program = SlangProgram::create(compile_context, kernel.composition);
    kernel.entry_point = SlangProgramEntryPoint::create(kernel.program, "main");
    kernel.pipeline = Versioned<Pipeline>([this, &kernel] {
        const auto ep = kernel.entry_point.get();
        return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
    });
    kernel.pipeline.depends_on(kernel.entry_point);
    kernel.globals = Versioned<ShaderObject>([this, &kernel] {
        return kernel.entry_point.get()->create_global_shader_object(context, allocator);
    });
    kernel.globals.depends_on(kernel.entry_point);
}

std::vector<InputConnectorDescriptor> PathDebugNode::describe_inputs() {
    con_gbuffer = GBufferIn::create({{{GBufferField::Hit}},
                                     {{GBufferField::Normal}},
                                     {{GBufferField::LinearZ}},
                                     {{GBufferField::GradZ}}});
    return {
        {"scene", con_scene},
        {"records", con_records,
         ConnectorAccess::compute_read_write |
             ConnectorAccess{vk::PipelineStageFlagBits2::eVertexShader,
                             vk::AccessFlagBits2::eShaderRead,
                             {},
                             {}}},
        {"src", con_src, ConnectorAccess::compute_read},
        {"backdrop", con_backdrop, ConnectorAccess::compute_read, 1, true},
        {"gbuffer", con_gbuffer, ConnectorAccess::compute_read | ConnectorAccess::fragment_read, 0,
         true},
        {"controller", con_controller, {}, 0, true},
        {"window", con_window, {}, 0, true},
        {"acquire", con_acquire, ConnectorAccess::transfer_dst, 0, true},
    };
}

std::vector<OutputConnectorDescriptor>
PathDebugNode::describe_outputs(const NodeIOLayout& io_layout) {
    grid_slots_log2 = std::clamp(grid_slots_log2, 8, 26);
    const vk::ImageCreateInfo src_info = io_layout[con_src]->get_create_info_or_throw();
    extent = src_info.extent;

    const vk::ImageCreateInfo out_info{
        {},
        vk::ImageType::e2D,
        src_info.format,
        extent,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };
    con_out = ManagedVkImageOut::create(out_info);
    con_filtered = ManagedVkImageOut::create(vk::Format::eR32G32B32A32Sfloat, extent);
    con_filtered_b = ManagedVkImageOut::create(vk::Format::eR32G32B32A32Sfloat, extent);

    const vk::DeviceSize pixels = vk::DeviceSize{extent.width} * extent.height;
    moments_buffer = allocator->create_buffer(pixels * 6 * 4,
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              MemoryMappingType::NONE, "path_debug moments");
    filtered_buffer = allocator->create_buffer(((pixels * 4 * PATH_DEBUG_SLOT_COUNT) + 4) * 4,
                                               vk::BufferUsageFlagBits::eStorageBuffer |
                                                   vk::BufferUsageFlagBits::eTransferDst,
                                               MemoryMappingType::NONE, "path_debug filtered");
    grid_buffer = allocator->create_buffer(
        (vk::DeviceSize{2} << static_cast<uint32_t>(grid_slots_log2)) * 4,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::NONE, "path_debug heat grid");
    map_res_log2 = std::clamp(map_res_log2, 3, 10);
    allocated_map_res = 1u << static_cast<uint32_t>(map_res_log2);
    const vk::DeviceSize map_bins =
        vk::DeviceSize{allocated_map_res} * vk::DeviceSize{allocated_map_res};
    maps_buffer = allocator->create_buffer(map_bins * PATH_DEBUG_MAP_REGIONS * 4,
                                           vk::BufferUsageFlagBits::eStorageBuffer |
                                               vk::BufferUsageFlagBits::eTransferDst |
                                               vk::BufferUsageFlagBits::eTransferSrc,
                                           MemoryMappingType::NONE, "path_debug maps");

    return {{"out", con_out,
             ConnectorAccess::compute_write | ConnectorAccess::color_attachment |
                 ConnectorAccess::transfer_src},
            {"filtered", con_filtered, ConnectorAccess::compute_write},
            {"filtered_b", con_filtered_b, ConnectorAccess::compute_write},
            {"acquire", con_acquire_out}};
}

PathDebugNode::NodeStatusFlags
PathDebugNode::on_connected(const NodeIOLayout& io_layout,
                            [[maybe_unused]] const NodeIO& io,
                            const NodeConnectionInfo& info,
                            [[maybe_unused]] Submission& submission) {
    gbuffer_connected = io_layout.is_connected(con_gbuffer);
    gbuffer_composition =
        (gbuffer_connected ? io[con_gbuffer]->get_layout() : GBufferLayout::complete())
            ->get_composition();
    build_gbuffer_kernels();
    bsdf_map = {};
    bsdf_check = {};
    maps_dirty = true;
    heat_dirty = true;

    const PathRecordCapacities capacities = path_record_capacities(io[con_records]->get_size());
    record_path_capacity = capacities.path_capacity;
    matches_buffer = allocator->create_buffer(
        std::max<vk::DeviceSize>(
            vk::DeviceSize{(capacities.path_capacity + 31) / 32} * 4 * PATH_DEBUG_SLOT_COUNT, 8),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::NONE, "path_debug matches");

    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            const bool forced = std::any_cast<bool>(force);
            for (auto* kernel :
                 {&select_kernel, &threshold_kernel, &collect_kernel, &stats_kernel,
                  &map_reduce_kernel, &map_view_kernel, &compose_kernel, &heat_decay_kernel,
                  &heat_splat_kernel, &filter_render_kernel, &finalize_kernel}) {
                (*kernel)->reload(forced, compile_context);
            }
            for (const SlangCompositionHandle& composition :
                 {bsdf_map.composition, bsdf_check.composition, overlay_composition}) {
                if (!composition) {
                    continue;
                }
                if (forced) {
                    composition->force_reload();
                } else {
                    composition->reload(compile_context->get_search_path_file_loader());
                }
            }
            pdf_stale = true;
            return true;
        });
    io_layout.register_event_listener("//camera_changed,//geometry_changed,//transform_changed",
                                      [this](const GraphEvent::Info&, const GraphEvent::Data&) {
                                          pixel_stats_dirty = true;
                                          maps_dirty = true;
                                          resampled_dirty = true;
                                          return false;
                                      });
    io_layout.register_event_listener("//geometry_changed,//transform_changed",
                                      [this](const GraphEvent::Info&, const GraphEvent::Data&) {
                                          heat_dirty = true;
                                          return false;
                                      });
    io_layout.register_event_listener(
        imgui_event_pattern, [this](const GraphEvent::Info&, const GraphEvent::Data& data) {
            if (Properties* const* props = std::any_cast<Properties*>(&data); props != nullptr) {
                draw_window();
            }
            return false;
        });

    readback_buffers.resize(info.iterations_in_flight);
    for (auto& buffer : readback_buffers) {
        buffer =
            allocator->create_buffer(sizeof(Readback), vk::BufferUsageFlagBits::eTransferDst,
                                     MemoryMappingType::HOST_ACCESS_RANDOM, "path_debug readback");
    }
    return {};
}

PathDebugNode::NodeStatusFlags
PathDebugNode::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    handle_input(io);
    load_reference(submission);

    const PathRecordCapacities capacities = path_record_capacities(io[con_records]->get_size());
    const ReadbackFeedback fed = apply_readback_feedback(info, capacities);
    const SceneHandle& scene = io[con_scene];
    const bool ready = scene && scene->is_ready();

    record_clears(io, info, submission);
    const FrameActivity activity = update_params(io, info, ready, fed);
    bind_globals(io);

    if (ready) {
        record_analysis(io, info, submission, capacities, activity);
    }
    record_compose(io, info, submission);
    if (ready) {
        record_overlay_lines(io, info, submission);
        record_map_panels(io, info, submission);
    }
    record_labels(io, info, submission);
    record_export(io, info, submission);
    if (ready) {
        record_readback(io, info, submission);
    }
    record_present(io, submission);
    const std::scoped_lock lock(stats_mutex);
    return std::exchange(pending_reconnect, false) ? NEEDS_RECONNECT : NodeStatusFlags{};
}

void PathDebugNode::handle_input(const NodeIO& io) {
    if (io.is_connected(con_controller)) {
        const InputControllerHandle& controller = io[con_controller];
        if (controller && controller != registered_controller.lock()) {
            controller->add_listener(input, PathDebugInput::PRIORITY);
            registered_controller = controller;
        }
    }
    if (input->take_clear()) {
        clear_selection();
    }

    if (const double scroll = input->take_scroll(); scroll != 0.) {
        const auto cursor = cursor_pixel();
        if (cursor && panel_at(*cursor)) {
            panel_scroll = std::max(0, panel_scroll - static_cast<int32_t>(scroll * 60.));
        } else if (picked_pos_valid) {
            picked_radius =
                std::max(1e-6f, picked_radius * std::pow(1.15f, static_cast<float>(scroll)));
            invalidate_selection();
        }
    }
    const auto clicked = input->take_click();
    if (const auto click = clicked ? to_image(*clicked) : std::nullopt) {
        const auto hit = panel_at(*click);
        if (hit && hit->on_close) {
            panel_enabled[hit->channel] = false;
        } else {
            picked_pixel = *click;
            picked = true;
            picked_pos_valid = false;
            picked_radius = 0.f;
            pending_pick_pixel = *click;
            pending_pick = true;
            pending_pick_seq++;
            invalidate_selection();
        }
    }

    params.cursor_x = PATH_DEBUG_NO_PIXEL;
    params.cursor_y = PATH_DEBUG_NO_PIXEL;
    params.probe_channel = PATH_DEBUG_MAP_NONE;
    if (const auto cursor = cursor_pixel()) {
        params.cursor_x = static_cast<uint32_t>(cursor->x);
        params.cursor_y = static_cast<uint32_t>(cursor->y);
        if (const auto hit = panel_at(*cursor); hit && !hit->on_close) {
            uint32_t slot = 0;
            for (uint32_t channel = 0; channel < hit->channel; channel++) {
                slot += (shown_panel_mask() >> channel) & 1u;
            }
            const uint32_t bx = static_cast<uint32_t>(cursor->x - panel_left()) * params.map_res /
                                params.map_panel_size;
            const uint32_t by = static_cast<uint32_t>(cursor->y - panel_top(slot)) *
                                params.map_res / params.map_panel_size;
            params.probe_channel = hit->channel;
            params.probe_bin = (std::min(by, params.map_res - 1) * params.map_res) +
                               std::min(bx, params.map_res - 1);
        }
    }
}

PathDebugNode::ReadbackFeedback
PathDebugNode::apply_readback_feedback(const NodeProcessInfo& info,
                                       const PathRecordCapacities& capacities) {
    if (info.get_iteration() == 0 && auto_keep_prob) {
        keep_prob = MIN_KEEP_PROB;
    }

    const std::scoped_lock lock(stats_mutex);
    const auto tick = [](bool& drawn, uint32_t& frames) {
        frames = std::exchange(drawn, false) ? 0 : std::min(frames + 1, UI_IDLE_FRAMES);
    };
    tick(window_drawn, frames_since_window);
    tick(inspector_drawn, frames_since_inspector);
    tick(convergence_drawn, frames_since_convergence);

    ReadbackFeedback fed{capacities.path_capacity, 0, std::exchange(check_requested, false)};
    if (!readback_valid) {
        return fed;
    }
    fed.observed_paths = std::min(latest_readback.frame.paths, capacities.path_capacity);
    fed.matched_paths = latest_readback.frame.matched;
    if (latest_readback.stats[PATH_DEBUG_STATS_COUNT] > (1u << 30)) {
        maps_dirty = true;
    }
    adapt_keep_prob(capacities);
    resolve_pick();
    return fed;
}

void PathDebugNode::adapt_keep_prob(const PathRecordCapacities& capacities) {
    if (!auto_keep_prob || freeze || (record_picked_pixel_only && picked)) {
        return;
    }
    const float vertex_fill = static_cast<float>(latest_readback.frame.vertices) /
                              static_cast<float>(capacities.vertex_capacity);
    const float path_fill = static_cast<float>(latest_readback.frame.paths) /
                            static_cast<float>(capacities.path_capacity);
    const float fill = std::max(vertex_fill, path_fill);
    const float keep_used = std::bit_cast<float>(latest_readback.frame.keep_prob);
    if (keep_used > 0.f && (fill > 1.f || fill < 0.5f)) {
        keep_prob = std::clamp(keep_used * 0.9f / std::max(fill, 1e-3f), MIN_KEEP_PROB, 1.f);
    }
}

void PathDebugNode::resolve_pick() {
    if (!pending_pick || latest_readback.pick.seq != pending_pick_seq) {
        return;
    }
    const float3 hit(std::bit_cast<float>(latest_readback.pick.x),
                     std::bit_cast<float>(latest_readback.pick.y),
                     std::bit_cast<float>(latest_readback.pick.z));
    if (length(hit) >= PATH_DEBUG_ENV_DISTANCE) {
        return;
    }
    picked_pos = hit;
    if (picked_radius <= 0.f) {
        picked_radius = 48.f * std::bit_cast<float>(latest_readback.pick.footprint);
    }
    picked_pos_valid = true;
    pending_pick = false;
    invalidate_selection();
}

void PathDebugNode::record_clears(const NodeIO& io,
                                  const NodeProcessInfo& info,
                                  Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    const BufferHandle& records = io[con_records];
    cmd->barrier({compute_to_transfer(records), compute_to_transfer(state_buffer),
                  compute_to_transfer(matches_buffer), compute_to_transfer(filtered_buffer),
                  compute_to_transfer(stats_buffer), compute_to_transfer(maps_buffer),
                  compute_to_transfer(grid_buffer), compute_to_transfer(moments_buffer),
                  compute_to_transfer(resampled_buffer)});
    if (info.get_iteration() == 0) {
        cmd->fill(records);
    }
    cmd->fill(state_buffer);
    cmd->fill(matches_buffer);
    if (!map_accumulate) {
        maps_dirty = true;
    }
    if (maps_dirty) {
        cmd->fill(stats_buffer);
        cmd->fill(maps_buffer);
        maps_dirty = false;
        map_frames = 0;
        pdf_stale = true;
    }
    if (heat_dirty || info.get_iteration() == 0) {
        cmd->fill(grid_buffer);
        heat_dirty = false;
        heat_frames = 0;
    }
    if (pixel_stats_dirty || info.get_iteration() == 0) {
        cmd->fill(moments_buffer);
        cmd->fill(filtered_buffer);
        pixel_stats_dirty = false;
        pixel_stats_frames = 0;
        const std::scoped_lock lock(stats_mutex);
        mean_history.clear();
        rel_error_history.clear();
        variance_history.clear();
    }
    if (resampled_dirty || info.get_iteration() == 0) {
        cmd->fill(resampled_buffer);
        resampled_dirty = false;
        resampled_frames = 0;
        const std::scoped_lock lock(stats_mutex);
        resampled_from = 0;
    }
    if (info.get_iteration() == 0) {
        cmd->barrier(transfer_to_compute(records));
    }
    cmd->barrier({transfer_to_compute(state_buffer), transfer_to_compute(stats_buffer),
                  transfer_to_compute(maps_buffer), transfer_to_compute(grid_buffer),
                  transfer_to_compute(moments_buffer), transfer_to_compute(filtered_buffer),
                  transfer_to_compute(matches_buffer), transfer_to_compute(resampled_buffer)});
}

PathDebugNode::FrameActivity PathDebugNode::update_params(const NodeIO& io,
                                                          const NodeProcessInfo& info,
                                                          const bool ready,
                                                          const ReadbackFeedback& fed) {
    picked_pixel.x = std::clamp(picked_pixel.x, 0, static_cast<int32_t>(extent.width) - 1);
    picked_pixel.y = std::clamp(picked_pixel.y, 0, static_cast<int32_t>(extent.height) - 1);
    params.dim_x = extent.width;
    params.dim_y = extent.height;

    const bool frozen = freeze;
    update_selection_params();
    update_capture_params();
    update_overlay_params(fed, frozen);
    update_map_params(fed);
    update_heat_params(frozen);
    update_pixel_stats_params(io, frozen, ready);
    update_backdrop_params(info);

    if (fed.run_check) {
        check_seq++;
    }
    params.check_seq = check_seq;
    const bool maps_active = panels_enabled || params.sphere_channel != PATH_DEBUG_MAP_NONE ||
                             frames_since_window < UI_IDLE_FRAMES;
    const bool maps_accumulate = maps_active && (!frozen || map_frames == 0);
    params.map_batch_end =
        maps_accumulate &&
                (map_frames + 1) % static_cast<uint32_t>(std::max(map_batch_frames, 1)) == 0
            ? 1
            : 0;
    return {frozen, maps_active, fed.run_check};
}

void PathDebugNode::update_selection_params() {
    if (picked && !picked_pos_valid && !pending_pick) {
        pending_pick_pixel = picked_pixel;
        pending_pick = true;
        pending_pick_seq++;
    }
    params.sel_pixel_x = picked ? static_cast<uint32_t>(picked_pixel.x) : PATH_DEBUG_NO_PIXEL;
    params.sel_pixel_y = picked ? static_cast<uint32_t>(picked_pixel.y) : PATH_DEBUG_NO_PIXEL;
    params.sel_region_x = picked_pos.x;
    params.sel_region_y = picked_pos.y;
    params.sel_region_z = picked_pos.z;
    params.sel_region_radius = picked_pos_valid ? std::max(picked_radius, 0.f) : 0.f;
    params.sel_restrict = static_cast<uint32_t>(restrict_mode);
    scatter_range.x = std::clamp(scatter_range.x, 0, 32);
    scatter_range.y = std::clamp(scatter_range.y, scatter_range.x, 32);
    params.sel_scatters_min = static_cast<uint32_t>(scatter_range.x);
    params.sel_scatters_max =
        scatter_range.y >= 32 ? 0xFFFFFFFFu : static_cast<uint32_t>(scatter_range.y);
    params.sel_method_mask = method_mask;
    params.sel_method_mode = static_cast<uint32_t>(method_mode);
    params.sel_finite = static_cast<uint32_t>(contribution_filter);
    params.sel_material =
        material_id >= 0 ? static_cast<uint32_t>(material_id) : PATH_RECORD_MATERIAL_NONE;
    params.expr_a_0 = compiled_a.words[0];
    params.expr_a_1 = compiled_a.words[1];
    params.expr_a_2 = compiled_a.words[2];
    params.expr_a_3 = compiled_a.words[3];
    params.expr_a_count =
        compiled_a.valid ? compiled_a.token_count | (compiled_a.nee ? PATH_DEBUG_EXPR_NEE : 0u) : 0;
    params.expr_b_0 = compiled_b.words[0];
    params.expr_b_1 = compiled_b.words[1];
    params.expr_b_2 = compiled_b.words[2];
    params.expr_b_3 = compiled_b.words[3];
    params.expr_b_count =
        compiled_b.valid ? compiled_b.token_count | (compiled_b.nee ? PATH_DEBUG_EXPR_NEE : 0u) : 0;
    params.pick_x =
        pending_pick ? static_cast<uint32_t>(pending_pick_pixel.x) : PATH_DEBUG_NO_PIXEL;
    params.pick_y =
        pending_pick ? static_cast<uint32_t>(pending_pick_pixel.y) : PATH_DEBUG_NO_PIXEL;
    params.pick_seq = pending_pick_seq;
}

void PathDebugNode::update_capture_params() {
    const bool one_pixel = record_picked_pixel_only && picked;
    params.freeze = freeze ? 1 : 0;
    params.keep_prob = one_pixel ? 1.f : keep_prob;
    params.record_pixel =
        one_pixel ? ((params.sel_pixel_y << 16) | params.sel_pixel_x) : PATH_RECORD_ALL_PIXELS;
}

void PathDebugNode::update_overlay_params(const ReadbackFeedback& fed, const bool frozen) {
    const bool resampling = overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_RESAMPLED);
    params.overlay_source = static_cast<uint32_t>(overlay_source);
    params.overlay_rank = static_cast<uint32_t>(overlay_rank);
    params.overlay_max_paths = std::min(static_cast<uint32_t>(std::max(overlay_max_paths, 1)),
                                        resampling ? PATH_DEBUG_RESAMPLED_SLOTS : MAX_DRAW);
    params.overlay_resample = resampling && (!frozen || resampled_frames == 0) ? 1 : 0;
    resampled_frames += params.overlay_resample;
    params.overlay_brightest_fraction = std::clamp(overlay_brightest_fraction, 1e-7f, 1.f);
    params.overlay_sample_max = std::numeric_limits<uint32_t>::max();
    if (overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_ANY) &&
        fed.matched_paths > params.overlay_max_paths) {
        const double keep =
            static_cast<double>(params.overlay_max_paths) / static_cast<double>(fed.matched_paths);
        params.overlay_sample_max =
            static_cast<uint32_t>(std::numeric_limits<uint32_t>::max() * keep);
    }
    params.overlay_depth_mode = static_cast<uint32_t>(overlay_depth_mode);
    params.overlay_color = static_cast<uint32_t>(overlay_color);
    params.overlay_isolate =
        isolate_path < 0 ? PATH_DEBUG_NO_ISOLATE : static_cast<uint32_t>(isolate_path);
}

void PathDebugNode::update_map_params(const ReadbackFeedback& fed) {
    params.map_source = static_cast<uint32_t>(map_source);
    params.map_res = allocated_map_res;
    params.map_transform = static_cast<uint32_t>(map_transform);
    params.map_frame =
        gbuffer_connected ? static_cast<uint32_t>(map_frame) : PATH_DEBUG_FRAME_WORLD;
    params.map_pdf_ref = static_cast<uint32_t>(map_pdf_ref);
    params.map_bounce = map_bounce < 0 ? PATH_DEBUG_ALL_BOUNCES : static_cast<uint32_t>(map_bounce);
    params.map_draws = static_cast<uint32_t>(map_draws);
    params.wi_mode = static_cast<uint32_t>(wi_mode);
    params.wi_theta = wi_theta_deg * std::numbers::pi_v<float> / 180.f;
    params.wi_phi = wi_phi_deg * std::numbers::pi_v<float> / 180.f;
    params.sphere_channel = (sphere_channel < 0 || !gbuffer_connected)
                                ? PATH_DEBUG_MAP_NONE
                                : static_cast<uint32_t>(sphere_channel);

    const uint32_t panel_limit = std::min(extent.width, extent.height);
    params.map_panel_size = std::max(
        1u, std::min(params.map_panel_size, panel_limit > 2 * PATH_DEBUG_PANEL_MARGIN
                                                ? panel_limit - (2 * PATH_DEBUG_PANEL_MARGIN)
                                                : panel_limit));
    params.map_panel_mask = panels_enabled ? shown_panel_mask() : 0;
    const auto panel_count = static_cast<uint32_t>(std::popcount(params.map_panel_mask));
    const auto column =
        static_cast<int32_t>((panel_count * (params.map_panel_size + PATH_DEBUG_PANEL_GAP)) +
                             (2 * PATH_DEBUG_PANEL_MARGIN));
    panel_overflow = std::max(0, column - static_cast<int32_t>(extent.height));
    panel_scroll = std::clamp(panel_scroll, 0, panel_overflow);
    params.map_panel_scroll = static_cast<uint32_t>(panel_scroll);
    params.want_top_list = frames_since_inspector < UI_IDLE_FRAMES ? 1 : 0;

    const bool pixel_restricted =
        picked && restrict_mode == static_cast<int32_t>(PATH_DEBUG_RESTRICT_PIXEL);
    params.stats_stride =
        (analysis_paths <= 0 || pixel_restricted)
            ? 1
            : std::max(fed.observed_paths / static_cast<uint32_t>(analysis_paths), 1u);
}

void PathDebugNode::update_heat_params(const bool frozen) {
    grid_slots_log2 = std::clamp(grid_slots_log2, 8, 26);
    params.heat_mode = gbuffer_connected ? static_cast<uint32_t>(heat_mode) : PATH_DEBUG_HEAT_OFF;
    params.heat_source = static_cast<uint32_t>(heat_source);
    heat_bounce_range.x = std::clamp(heat_bounce_range.x, 0, 30);
    heat_bounce_range.y = std::clamp(heat_bounce_range.y, heat_bounce_range.x, 30);
    params.heat_bounce_min = static_cast<uint32_t>(heat_bounce_range.x);
    params.heat_bounce_max = static_cast<uint32_t>(heat_bounce_range.y);
    const bool heat_splat = !frozen || heat_frames == 0;
    params.heat_decay = heat_splat ? 1.f - heat_alpha : 1.f;
    params.grid_slots = static_cast<uint32_t>(grid_buffer->get_size() / 4 / 2);
}

void PathDebugNode::update_pixel_stats_params(const NodeIO& io,
                                              const bool frozen,
                                              const bool ready) {
    const bool active = pixel_stats_view != static_cast<int32_t>(PATH_DEBUG_PIXEL_STAT_OFF) ||
                        params.cursor_x != PATH_DEBUG_NO_PIXEL ||
                        frames_since_convergence < UI_IDLE_FRAMES;
    params.pixel_stats_source = static_cast<uint32_t>(pixel_stats_source);
    params.pixel_stats_accumulate = (active && (!frozen || pixel_stats_frames == 0)) ? 1 : 0;
    params.pixel_stats_view = static_cast<uint32_t>(pixel_stats_view);
    if (params.pixel_stats_accumulate != 0 && ready) {
        pixel_stats_frames++;
    }
    const bool shows_filtered =
        backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_FILTERED_A) ||
        backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_FILTERED_B) ||
        backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_SPLIT_FILTERED);
    params.filtered_active =
        (shows_filtered || io.is_connected(con_filtered) || io.is_connected(con_filtered_b)) ? 1
                                                                                             : 0;
}

void PathDebugNode::update_backdrop_params(const NodeProcessInfo& info) {
    params.backdrop = static_cast<uint32_t>(backdrop);
    params.reference_ready = reference_loaded ? 1 : 0;
    if (reference_flip && (backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_REFERENCE) ||
                           backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_SPLIT_REFERENCE))) {
        const auto phase = static_cast<uint64_t>(info.get_elapsed() * reference_flip_hz);
        params.backdrop =
            (phase & 1u) == 0 ? PATH_DEBUG_BACKDROP_RENDER : PATH_DEBUG_BACKDROP_REFERENCE;
    }
}

void PathDebugNode::bind_node_buffers(ShaderCursor cursor, const NodeIO& io) const {
    const std::array<std::pair<const char*, BufferHandle>, 15> buffers = {{
        {"state", state_buffer},
        {"out_state", out_state_buffer},
        {"stats", stats_buffer},
        {"stats_f", stats_buffer},
        {"maps", maps_buffer},
        {"maps_f", maps_buffer},
        {"grid", grid_buffer},
        {"grid_f", grid_buffer},
        {"moments", moments_buffer},
        {"moments_f", moments_buffer},
        {"filtered", filtered_buffer},
        {"filtered_f", filtered_buffer},
        {"matches", matches_buffer},
        {"resampled", resampled_buffer},
        {"in_records", io[con_records]},
    }};
    for (const auto& [name, buffer] : buffers) {
        if (auto c = cursor.find(name); c.is_valid()) {
            c = buffer;
        }
    }
    if (auto c = cursor.find("in_reference"); c.is_valid() && reference_texture) {
        c = reference_texture;
    }
    if (auto c = cursor.find("in_backdrop"); c.is_valid()) {
        c = io.is_connected(con_backdrop) ? io[con_backdrop].get_texture()
                                          : io[con_src].get_texture();
    }
    if (auto c = cursor.find("params"); c.is_valid()) {
        c = params;
    }
    const SceneHandle& scene = io[con_scene];
    if (auto c = cursor.find("camera"); c.is_valid() && scene && scene->is_ready()) {
        scene->get_active_camera()->write_to(c);
    }
    if (auto c = cursor.find("gbuffer"); c.is_valid() && gbuffer_connected) {
        c = io[con_gbuffer].r();
    }
}

void PathDebugNode::bind_globals(const NodeIO& io) {
    for (auto* kernel : {&select_kernel, &threshold_kernel, &collect_kernel, &stats_kernel,
                         &map_reduce_kernel, &map_view_kernel, &compose_kernel, &heat_decay_kernel,
                         &heat_splat_kernel, &finalize_kernel, &filter_render_kernel}) {
        bind_node_buffers((*kernel)->globals_cursor(), io);
    }
}

void PathDebugNode::record_scene_kernel(ScenePipeline& kernel,
                                        const NodeIO& io,
                                        const NodeProcessInfo& info,
                                        Submission& submission,
                                        const vk::Extent3D& threads) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    const SceneHandle& scene = io[con_scene];
    const auto ep = kernel.entry_point.get();
    const auto pipe = kernel.pipeline.get();
    const auto globals = kernel.globals.get();
    bind_node_buffers(globals->get_cursor(), io);
    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, info.get_shader_object_allocator());
    ep->bind_global(globals, cmd, pipe, info.get_shader_object_allocator());
    cmd->dispatch(threads, 16, 16);
}

void PathDebugNode::record_analysis(const NodeIO& io,
                                    const NodeProcessInfo& info,
                                    Submission& submission,
                                    const PathRecordCapacities& capacities,
                                    const FrameActivity& activity) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    const SceneHandle& scene = io[con_scene];
    const uint32_t path_groups = groups(capacities.path_capacity);
    const bool heat_splat = !activity.frozen || heat_frames == 0;
    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "select");
        const auto pipe = select_kernel->bind(io, info, submission);
        cmd->dispatch(path_groups, 1, 1);
    }
    cmd->barrier(compute_to_compute(matches_buffer));
    if (params.filtered_active != 0) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "filter render");
        const auto pipe = filter_render_kernel->bind(io, info, submission);
        cmd->dispatch(path_groups, 1, 1);
    }
    if (activity.maps_active && (!activity.frozen || map_frames == 0)) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "stats");
        const uint32_t stats_threads =
            (capacities.path_capacity + params.stats_stride - 1) / params.stats_stride;
        const auto pipe = stats_kernel->bind(io, info, submission);
        cmd->dispatch(groups(stats_threads), 1, 1);
        map_frames++;
    }
    if (pdf_stale && activity.maps_active && gbuffer_connected) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "bsdf map");
        ensure_scene_pipeline(bsdf_map, scene, BSDF_MAP_MODULE);
        record_scene_kernel(bsdf_map, io, info, submission,
                            vk::Extent3D{params.map_res, params.map_res, 1});
        pdf_stale = false;
    }
    if (activity.run_check && gbuffer_connected) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "bsdf check");
        const std::array<const uint32_t, PATH_DEBUG_CHECK_UINTS> zero{};
        cmd->update(stats_buffer, sizeof(uint32_t) * PATH_DEBUG_CHECK,
                    vk::ArrayProxy<const uint32_t>(zero));
        cmd->barrier(transfer_to_compute(stats_buffer));
        ensure_scene_pipeline(bsdf_check, scene, BSDF_CHECK_MODULE);
        record_scene_kernel(bsdf_check, io, info, submission,
                            vk::Extent3D{PATH_DEBUG_CHECK_DIM, PATH_DEBUG_CHECK_DIM, 1});
    }
    cmd->barrier({compute_to_compute(state_buffer), compute_to_compute(stats_buffer),
                  compute_to_compute(maps_buffer)});
    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "reduce+threshold");
        if (activity.maps_active) {
            const auto reduce_pipe = map_reduce_kernel->bind(io, info, submission);
            cmd->dispatch(groups(params.map_res * params.map_res), 1, 1);
            cmd->barrier(compute_to_compute(stats_buffer));
        }
        const auto pipe = threshold_kernel->bind(io, info, submission);
        cmd->dispatch(1, 1, 1);
    }
    cmd->barrier({compute_to_compute(state_buffer), compute_to_compute(stats_buffer)});
    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "collect");
        const auto collect_pipe = collect_kernel->bind(io, info, submission);
        cmd->dispatch(path_groups, 1, 1);
    }
    if (params.heat_mode != PATH_DEBUG_HEAT_OFF) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "heat");
        {
            const auto pipe = heat_decay_kernel->bind(io, info, submission);
            cmd->dispatch(params.grid_slots / PATH_DEBUG_WG, 1, 1);
        }
        if (heat_splat) {
            cmd->barrier(compute_to_compute(grid_buffer));
            const auto pipe = heat_splat_kernel->bind(io, info, submission);
            cmd->dispatch(path_groups, 1, 1);
            heat_frames++;
        }
    }
    cmd->barrier({compute_to_compute(stats_buffer), compute_to_compute(maps_buffer),
                  compute_to_compute(grid_buffer), compute_to_compute(state_buffer),
                  compute_to_compute(moments_buffer), compute_to_compute(filtered_buffer),
                  compute_to_compute(resampled_buffer)});
}

void PathDebugNode::record_compose(const NodeIO& io,
                                   const NodeProcessInfo& info,
                                   Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "compose");
    const auto pipe = compose_kernel->bind(io, info, submission);
    cmd->dispatch(extent, 16, 16);
}

void PathDebugNode::record_overlay_lines(const NodeIO& io,
                                         const NodeProcessInfo& info,
                                         Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    const BufferHandle& records = io[con_records];
    MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "lines");
    if (overlay_format != io[con_out]->get_format()) {
        build_overlay_pipeline(io[con_out]->get_format());
    }
    const auto vertex = overlay_vertex.get();
    const auto pipe = overlay_pipeline.get();
    const auto globals = overlay_globals.get();
    bind_node_buffers(globals->get_cursor(), io);

    cmd->barrier(
        {state_buffer->buffer_barrier2(
             vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eVertexShader,
             vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eShaderRead),
         resampled_buffer->buffer_barrier2(
             vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eVertexShader,
             vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eShaderRead),
         records->buffer_barrier2(vk::PipelineStageFlagBits2::eComputeShader |
                                      vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
                                  vk::PipelineStageFlagBits2::eVertexShader,
                                  vk::AccessFlagBits2::eShaderWrite,
                                  vk::AccessFlagBits2::eShaderRead)});
    cmd->barrier(io[con_out]->barrier2(
        vk::ImageLayout::eColorAttachmentOptimal, vk::AccessFlagBits2::eShaderWrite,
        vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput));
    const vk::Extent2D area{extent.width, extent.height};
    const vk::RenderingAttachmentInfo color{*io[con_out].get_texture(0)->get_view(),
                                            vk::ImageLayout::eColorAttachmentOptimal,
                                            vk::ResolveModeFlagBits::eNone,
                                            {},
                                            {},
                                            vk::AttachmentLoadOp::eLoad,
                                            vk::AttachmentStoreOp::eStore,
                                            {}};
    // the globals upload and its barriers have to happen outside the rendering instance
    cmd->bind(pipe);
    vertex->bind_global(globals, cmd, pipe, info.get_shader_object_allocator());

    cmd->begin_rendering(vk::RenderingInfo{{}, vk::Rect2D{{0, 0}, area}, 1, 0, color});
    cmd->set_viewport(vk::Viewport{0.f, 0.f, static_cast<float>(extent.width),
                                   static_cast<float>(extent.height), 0.f, 1.f});
    cmd->set_scissor(vk::Rect2D{{0, 0}, area});
    cmd->draw(6, 2 * params.overlay_max_paths * MAX_SEGMENTS);
    cmd->end_rendering();
    cmd->barrier(io[con_out]->barrier2(
        vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eComputeShader));
}

void PathDebugNode::record_map_panels(const NodeIO& io,
                                      const NodeProcessInfo& info,
                                      Submission& submission) {
    if (params.map_panel_mask == 0) {
        return;
    }
    const CommandBufferHandle& cmd = submission.get_cmd();
    MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "map panels");
    cmd->barrier(io[con_out]->barrier2(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
                                       vk::AccessFlagBits2::eShaderWrite,
                                       vk::PipelineStageFlagBits2::eComputeShader,
                                       vk::PipelineStageFlagBits2::eComputeShader));
    const auto pipe = map_view_kernel->bind(io, info, submission);
    cmd->dispatch(vk::Extent3D{params.map_panel_size, params.map_panel_size,
                               static_cast<uint32_t>(std::popcount(params.map_panel_mask))},
                  16, 16);
}

void PathDebugNode::record_labels(const NodeIO& io,
                                  const NodeProcessInfo& info,
                                  Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "labels");
    cmd->barrier(io[con_out]->barrier2(
        vk::ImageLayout::eColorAttachmentOptimal, vk::AccessFlagBits2::eShaderWrite,
        vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput));
    imgui_ctx->get_io().DisplaySize =
        ImVec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
    imgui_backend->new_frame(static_cast<float>(frametime.seconds()));
    frametime.reset();
    imgui_ctx->with_context([&] { draw_overlay(); });
    imgui_renderer->render(cmd, io[con_out].get_texture(0)->get_view());
    cmd->barrier(io[con_out]->barrier2(
        vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eAllCommands));
}

void PathDebugNode::record_export(const NodeIO& io,
                                  const NodeProcessInfo& info,
                                  Submission& submission) {
    if (!std::exchange(export_next, false)) {
        return;
    }
    submission.get_cmd()->barrier(io[con_out]->barrier2(
        vk::ImageLayout::eTransferSrcOptimal, {}, vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eTransfer));
    image_export(allocator, submission, info.get_profiler(), io[con_out],
                 vk::Extent2D{extent.width, extent.height},
                 std::filesystem::absolute(
                     export_path + image_format_extension(IMAGE_EXPORT_FORMATS.at(export_format))),
                 IMAGE_EXPORT_FORMATS.at(export_format),
                 export_metadata ? info.get_metadata() : ImageMetadata{}, !export_keep_alpha);
}

void PathDebugNode::record_readback(const NodeIO& io,
                                    const NodeProcessInfo& info,
                                    Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "readback");
    const BufferHandle& records = io[con_records];
    // the stats are copied before finalize zeroes their maxima
    const BufferHandle readback = readback_buffers[info.get_in_flight_index()];
    cmd->barrier(stats_buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eTransferRead));
    cmd->copy(stats_buffer, readback,
              vk::BufferCopy{0, offsetof(Readback, stats), sizeof(uint32_t) * STATS_UINTS});
    cmd->barrier({compute_to_compute(records), compute_to_compute(state_buffer),
                  stats_buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                                                vk::PipelineStageFlagBits2::eComputeShader,
                                                vk::AccessFlagBits2::eTransferRead,
                                                vk::AccessFlagBits2::eShaderWrite)});
    {
        const auto pipe = finalize_kernel->bind(io, info, submission);
        cmd->dispatch(1, 1, 1);
    }
    cmd->barrier(out_state_buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eTransferRead));
    cmd->copy(out_state_buffer, readback, vk::BufferCopy{0, 0, sizeof(uint32_t) * OUT_UINTS});

    cmd->barrier(readback->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eHost,
        vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eHostRead));
    submission.sync_to_cpu([this, readback, resampled = params.overlay_resample != 0]() {
        const Readback data = *readback->get_memory()->map_as<Readback>();
        readback->get_memory()->unmap();
        const std::scoped_lock lock(stats_mutex);
        latest_readback = data;
        if (resampled) {
            resampled_from += data.frame.matched;
        }
        readback_valid = true;
        mean_history.push_back(std::bit_cast<float>(data.probe.picked_mean));
        rel_error_history.push_back(std::bit_cast<float>(data.probe.picked_rel_error));
        variance_history.push_back(std::bit_cast<float>(data.probe.picked_variance));
        while (mean_history.size() > 512) {
            mean_history.pop_front();
            rel_error_history.pop_front();
            variance_history.pop_front();
        }
    });
}

void PathDebugNode::record_present(const NodeIO& io, Submission& submission) {
    presenting = false;
    if (!io.is_connected(con_acquire)) {
        return;
    }
    const std::shared_ptr<SwapchainAcquireResult>& acquire = io[con_acquire];
    io[con_acquire_out] = acquire;
    if (!acquire) {
        return;
    }

    const CommandBufferHandle& cmd = submission.get_cmd();
    const ImageHandle& src = io[con_out].get_image();
    const ImageHandle dst = acquire->image_view->get_image();
    cmd->barrier(dst->barrier2(vk::ImageLayout::eTransferDstOptimal, true));
    cmd->barrier(
        src->barrier2(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eMemoryWrite,
                      vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eAllCommands,
                      vk::PipelineStageFlagBits2::eTransfer));

    const vk::Filter filter =
        src->format_features() & vk::FormatFeatureFlagBits::eSampledImageFilterLinear
            ? vk::Filter::eLinear
            : vk::Filter::eNearest;
    cmd_blit_fit(cmd, src, vk::ImageLayout::eTransferSrcOptimal, src->get_extent(), dst,
                 vk::ImageLayout::eTransferDstOptimal, dst->get_extent(), vk::ClearColorValue{},
                 filter);

    std::tie(present_lower, present_upper) = fit(vk::Offset3D{}, to_offset(src->get_extent()),
                                                 vk::Offset3D{}, to_offset(dst->get_extent()));
    present_density =
        io.is_connected(con_window) && io[con_window] ? io[con_window]->get_pixel_density() : 1.f;
    presenting = true;
}

void PathDebugNode::load_reference(Submission& submission) {
    if (!reference_dirty && reference_texture) {
        return;
    }
    reference_dirty = false;
    const TextureHandle loaded = reference_path.empty()
                                     ? nullptr
                                     : image_load_texture(allocator, submission, reference_path);
    reference_loaded = loaded != nullptr;
    if (loaded) {
        reference_texture = loaded;
        return;
    }
    const std::array<float, 4> black{0.f, 0.f, 0.f, 1.f};
    reference_texture = allocator->create_texture_from_rgba32f(
        submission.get_cmd(), black.data(), 1, 1, vk::SamplerAddressMode::eClampToEdge,
        vk::Filter::eLinear, vk::Filter::eLinear, "path_debug reference");
    submission.get_cmd()->barrier(
        reference_texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
}

void PathDebugNode::invalidate_selection() {
    maps_dirty = true;
    pixel_stats_dirty = true;
    heat_dirty = true;
    resampled_dirty = true;
}

void PathDebugNode::clear_selection() {
    picked = false;
    picked_pos_valid = false;
    picked_radius = 0.f;
    pending_pick = false;
    isolate_path = -1;
    invalidate_selection();
}

void PathDebugNode::set_expression(const bool slot_b, const std::string& text) {
    (slot_b ? expr_b : expr_a) = text;
    (slot_b ? compiled_b : compiled_a) = compile_path_expression(text);
    invalidate_selection();
}

bool PathDebugNode::shared_support(const bool needs_camera_wi) const {
    return picked && restrict_mode == static_cast<int32_t>(PATH_DEBUG_RESTRICT_PIXEL) &&
           map_bounce == 0 &&
           (!needs_camera_wi || wi_mode == static_cast<int32_t>(PATH_DEBUG_WI_CAMERA));
}

uint32_t PathDebugNode::shown_panel_mask() const {
    uint32_t mask = 0;
    for (uint32_t channel = 0; channel < MAP_CHANNEL_COUNT; channel++) {
        mask |= panel_enabled[channel] ? (1u << channel) : 0;
    }
    return picked ? mask : mask & ~PATH_DEBUG_MAP_SURFACE_MASK;
}

int32_t PathDebugNode::panel_left() const {
    return static_cast<int32_t>(extent.width - params.map_panel_size - PATH_DEBUG_PANEL_MARGIN);
}

int32_t PathDebugNode::panel_top(const uint32_t slot) const {
    return static_cast<int32_t>(PATH_DEBUG_PANEL_MARGIN +
                                (slot * (params.map_panel_size + PATH_DEBUG_PANEL_GAP))) -
           panel_scroll;
}

std::optional<PathDebugNode::PanelHit> PathDebugNode::panel_at(const int2& pixel) const {
    const int32_t left = panel_left();
    const auto size = static_cast<int32_t>(params.map_panel_size);
    if (!panels_enabled || pixel.x < left || pixel.x >= left + size) {
        return std::nullopt;
    }
    const uint32_t mask = shown_panel_mask();
    uint32_t slot = 0;
    for (uint32_t channel = 0; channel < MAP_CHANNEL_COUNT; channel++) {
        if ((mask & (1u << channel)) == 0) {
            continue;
        }
        const int32_t top = panel_top(slot);
        if (pixel.y >= top && pixel.y < top + size) {
            const auto close = static_cast<int32_t>(PATH_DEBUG_PANEL_CLOSE);
            return PanelHit{channel, pixel.y < top + close && pixel.x >= left + size - close};
        }
        slot++;
    }
    return std::nullopt;
}

std::optional<int2> PathDebugNode::cursor_pixel() const {
    if (const auto at = input->cursor(); at && !mouse_over_ui) {
        return to_image(*at);
    }
    return std::nullopt;
}

std::optional<int2> PathDebugNode::to_image(const float2& window_pos) const {
    if (!presenting) {
        return std::nullopt;
    }
    const float width = static_cast<float>(present_upper.x - present_lower.x);
    const float height = static_cast<float>(present_upper.y - present_lower.y);
    if (width <= 0.f || height <= 0.f) {
        return std::nullopt;
    }
    const float u =
        ((window_pos.x * present_density) - static_cast<float>(present_lower.x)) / width;
    const float v =
        ((window_pos.y * present_density) - static_cast<float>(present_lower.y)) / height;
    if (u < 0.f || v < 0.f || u >= 1.f || v >= 1.f) {
        return std::nullopt;
    }
    return int2(static_cast<int32_t>(u * static_cast<float>(extent.width)),
                static_cast<int32_t>(v * static_cast<float>(extent.height)));
}

} // namespace merian
