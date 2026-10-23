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

namespace merian {

namespace {
constexpr const char* SELECT_MODULE = "merian-graph/nodes/path_debug/select.slang";
constexpr const char* THRESHOLD_MODULE = "merian-graph/nodes/path_debug/threshold.slang";
constexpr const char* COLLECT_MODULE = "merian-graph/nodes/path_debug/collect.slang";
constexpr const char* KEEP_MODULE = "merian-graph/nodes/path_debug/keep.slang";
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

constexpr std::array<const char*, 13> STANDALONE_MODULES = {
    SELECT_MODULE,   THRESHOLD_MODULE,     COLLECT_MODULE,  KEEP_MODULE,       COMPOSE_MODULE,
    STATS_MODULE,    MAP_REDUCE_MODULE,    MAP_VIEW_MODULE, HEAT_DECAY_MODULE, HEAT_SPLAT_MODULE,
    FINALIZE_MODULE, FILTER_RENDER_MODULE, OVERLAY_MODULE};

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

vk::BufferMemoryBarrier2 compute_to_vertex(const BufferHandle& buffer) {
    return buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eComputeShader |
                                       vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
                                   vk::PipelineStageFlagBits2::eVertexShader,
                                   vk::AccessFlagBits2::eShaderWrite,
                                   vk::AccessFlagBits2::eShaderRead);
}

vk::ImageMemoryBarrier2 to_color_attachment(const ImageHandle& image) {
    return image->barrier2(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                           vk::AccessFlagBits2::eColorAttachmentRead |
                               vk::AccessFlagBits2::eColorAttachmentWrite,
                           vk::PipelineStageFlagBits2::eAllCommands,
                           vk::PipelineStageFlagBits2::eColorAttachmentOutput);
}

vk::ImageMemoryBarrier2 from_color_attachment(const ImageHandle& image) {
    return image->barrier2(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eColorAttachmentWrite,
                           vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                           vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                           vk::PipelineStageFlagBits2::eAllCommands);
}

int2 to_int2(const vk::Extent3D& extent) {
    return {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height)};
}

uint32_t ceil_div(const uint32_t value, const uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

uint32_t groups(const uint32_t threads) {
    return ceil_div(threads, PATH_DEBUG_WG);
}
BufferHandle create_uint_buffer(const ResourceAllocatorHandle& allocator,
                                const vk::DeviceSize uints,
                                const vk::BufferUsageFlags usage,
                                const char* name) {
    return allocator->create_buffer(uints * sizeof(uint32_t),
                                    vk::BufferUsageFlagBits::eStorageBuffer | usage,
                                    MemoryMappingType::NONE, name);
}

} // namespace

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
    this->compile_context = context->get_shader_compile_context();

    spec_info.set(SpecializationInfoBuilder().build());

    select_kernel.emplace(context, allocator, compile_context, SELECT_MODULE, spec_info);
    threshold_kernel.emplace(context, allocator, compile_context, THRESHOLD_MODULE, spec_info);
    collect_kernel.emplace(context, allocator, compile_context, COLLECT_MODULE, spec_info);
    keep_kernel.emplace(context, allocator, compile_context, KEEP_MODULE, spec_info);
    map_reduce_kernel.emplace(context, allocator, compile_context, MAP_REDUCE_MODULE, spec_info);
    map_view_kernel.emplace(context, allocator, compile_context, MAP_VIEW_MODULE, spec_info);
    heat_decay_kernel.emplace(context, allocator, compile_context, HEAT_DECAY_MODULE, spec_info);
    heat_splat_kernel.emplace(context, allocator, compile_context, HEAT_SPLAT_MODULE, spec_info);
    finalize_kernel.emplace(context, allocator, compile_context, FINALIZE_MODULE, spec_info);
    filter_render_kernel.emplace(context, allocator, compile_context, FILTER_RENDER_MODULE,
                                 spec_info);
    build_gbuffer_kernels(context, allocator);

    state_buffer = create_uint_buffer(allocator, vk::DeviceSize{PATH_DEBUG_STATE_UINTS},
                                      vk::BufferUsageFlagBits::eTransferDst, "path_debug state");
    kept_buffer = create_uint_buffer(allocator, vk::DeviceSize{PATH_DEBUG_KEPT_UINTS},
                                     vk::BufferUsageFlagBits::eTransferDst, "path_debug kept");
    out_state_buffer =
        create_uint_buffer(allocator, vk::DeviceSize{PATH_DEBUG_OUT_UINTS},
                           vk::BufferUsageFlagBits::eTransferSrc, "path_debug out state");
    stats_buffer = create_uint_buffer(allocator, vk::DeviceSize{PATH_DEBUG_STATS_UINTS},
                                      vk::BufferUsageFlagBits::eTransferDst |
                                          vk::BufferUsageFlagBits::eTransferSrc,
                                      "path_debug stats");
    ::ImGuiContext* const prev_imgui_ctx = ImGui::GetCurrentContext();
    imgui_ctx = std::make_shared<ImGuiContext>();
    // the window's cursor and imgui.ini belong to a graph UI drawn over the canvas
    imgui_ctx->get_io().ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
    imgui_ctx->get_io().IniFilename = nullptr;
    imgui_renderer = std::make_shared<ImGuiRenderer>(context, allocator, imgui_ctx);
    imgui_backend = std::make_shared<ImGuiMerianBackend>(imgui_ctx);
    ImGuiContext::set_current(prev_imgui_ctx);
}

void PathDebugNode::build_gbuffer_kernels(const ContextHandle& context,
                                          const ResourceAllocatorHandle& allocator) {
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
    overlay_out = {};
    overlay_canvas = {};
}

void PathDebugNode::ensure_overlay_target(const ContextHandle& context,
                                          const ResourceAllocatorHandle& allocator,
                                          OverlayTarget& target,
                                          const vk::Format color_format) {
    if (target.format == color_format) {
        return;
    }
    target.format = color_format;
    target.globals = Versioned<ShaderObject>([this, context, allocator] {
        return overlay_vertex.get()->create_global_shader_object(context, allocator);
    });
    target.globals.depends_on(overlay_vertex);
    target.pipeline = Versioned<Pipeline>([this, context, color_format] {
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
    target.pipeline.depends_on(overlay_vertex);
    target.pipeline.depends_on(overlay_fragment);
}

void PathDebugNode::ensure_scene_pipeline(const ContextHandle& context,
                                          const ResourceAllocatorHandle& allocator,
                                          ScenePipeline& kernel,
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
    kernel.pipeline = Versioned<Pipeline>([&kernel, context] {
        const auto ep = kernel.entry_point.get();
        return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
    });
    kernel.pipeline.depends_on(kernel.entry_point);
    kernel.globals = Versioned<ShaderObject>([&kernel, context, allocator] {
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
         ConnectorAccess::compute_read_write | ConnectorAccess::transfer_dst |
             ConnectorAccess{vk::PipelineStageFlagBits2::eVertexShader,
                             vk::AccessFlagBits2::eShaderRead,
                             {},
                             {}}},
        {"src", con_src, ConnectorAccess::compute_read},
        {"backdrop", con_backdrop, ConnectorAccess::compute_read, 1, true},
        {"reference", con_reference, ConnectorAccess::compute_read, 0, true},
        {"gbuffer", con_gbuffer, ConnectorAccess::compute_read | ConnectorAccess::fragment_read, 0,
         true},
        {"controller", con_controller, {}, 0, true},
        {"window", con_window, {}, 0, true},
        {.name = "canvas",
         .connector = con_canvas,
         .access = ConnectorAccess::transfer_dst | ConnectorAccess::color_attachment,
         .optional = true,
         .in_place = true},
    };
}

std::vector<OutputConnectorDescriptor>
PathDebugNode::describe_outputs(const NodeIOLayout& io_layout) {
    grid_slots_log2 = std::clamp(grid_slots_log2, GRID_SLOTS_LOG2_MIN, GRID_SLOTS_LOG2_MAX);
    const vk::ImageCreateInfo src_info = io_layout[con_src]->get_create_info_or_throw();
    extent = io_layout.is_connected(con_gbuffer) ? io_layout[con_gbuffer]->get_create_info().extent
                                                 : src_info.extent;
    out_extent = src_info.extent;

    const vk::ImageCreateInfo out_info{
        {},
        vk::ImageType::e2D,
        src_info.format,
        out_extent,
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

    const vk::DeviceSize pixels = vk::DeviceSize{extent.width} * extent.height;
    const ResourceAllocatorHandle& allocator = io_layout.get_allocator();
    moments_buffer =
        create_uint_buffer(allocator, pixels * PATH_DEBUG_MOMENT_UINTS,
                           vk::BufferUsageFlagBits::eTransferDst, "path_debug moments");
    filtered_buffer =
        create_uint_buffer(allocator, ((pixels * PATH_DEBUG_FILTERED_UINTS) + 1),
                           vk::BufferUsageFlagBits::eTransferDst, "path_debug filtered");
    grid_buffer = create_uint_buffer(allocator,
                                     vk::DeviceSize{PATH_DEBUG_GRID_REGIONS}
                                         << static_cast<uint32_t>(grid_slots_log2),
                                     vk::BufferUsageFlagBits::eTransferDst, "path_debug heat grid");
    map_res_log2 = std::clamp(map_res_log2, MAP_RES_LOG2_MIN, MAP_RES_LOG2_MAX);
    allocated_map_res = 1u << static_cast<uint32_t>(map_res_log2);
    const vk::DeviceSize map_bins =
        vk::DeviceSize{allocated_map_res} * vk::DeviceSize{allocated_map_res};
    maps_buffer = create_uint_buffer(allocator, map_bins * PATH_DEBUG_MAP_REGIONS,
                                     vk::BufferUsageFlagBits::eTransferDst |
                                         vk::BufferUsageFlagBits::eTransferSrc,
                                     "path_debug maps");
    const ImageHandle atlas = allocator->create_image(
        vk::ImageCreateInfo{{},
                            vk::ImageType::e2D,
                            vk::Format::eR16G16B16A16Sfloat,
                            {allocated_map_res * PATH_DEBUG_PANEL_ATLAS_COLUMNS,
                             allocated_map_res * PATH_DEBUG_PANEL_ATLAS_ROWS, 1},
                            1,
                            1,
                            vk::SampleCountFlagBits::e1,
                            vk::ImageTiling::eOptimal,
                            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                            vk::SharingMode::eExclusive,
                            {},
                            {},
                            vk::ImageLayout::eUndefined},
        MemoryMappingType::NONE, "path_debug panels");
    panel_texture = allocator->create_texture(
        atlas, atlas->make_view_create_info(),
        allocator->get_sampler_pool()->nearest_clamp_to_edge(), "path_debug panels");

    return {{"out", con_out,
             ConnectorAccess::compute_write | ConnectorAccess::color_attachment |
                 ConnectorAccess::transfer_src},
            {"filtered", con_filtered, ConnectorAccess::compute_write}};
}

PathDebugNode::NodeStatusFlags PathDebugNode::on_connected(const NodeIOLayout& io_layout,
                                                           const NodeIO& io,
                                                           const NodeConnectionInfo& info,
                                                           Submission& submission) {
    gbuffer_connected = io_layout.is_connected(con_gbuffer);
    gbuffer_composition =
        (gbuffer_connected ? io[con_gbuffer]->get_layout() : GBufferLayout::complete())
            ->get_composition();
    const ResourceAllocatorHandle& allocator = io.get_allocator();
    build_gbuffer_kernels(io.get_context(), allocator);
    bsdf_map = {};
    bsdf_check = {};
    maps_dirty = true;
    heat_dirty = true;

    submission.get_cmd()->barrier(
        panel_texture->get_image()->barrier2(vk::ImageLayout::eGeneral, true));
    imgui_renderer->unregister_texture(panel_texture_id);
    panel_texture_id = imgui_renderer->register_texture(panel_texture);

    const PathRecordCapacities capacities = path_record_capacities(io[con_records]->get_size());
    record_path_capacity = capacities.path_capacity;
    matches_buffer = create_uint_buffer(
        allocator,
        std::max<vk::DeviceSize>(vk::DeviceSize{(capacities.path_capacity + 31) / 32}, 2),
        vk::BufferUsageFlagBits::eTransferDst, "path_debug matches");

    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            const bool forced = std::any_cast<bool>(force);
            for (std::optional<ComputeKernel>* const kernel : kernels()) {
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
            maps_dirty = true;
        });
    io_layout.register_event_listener(
        "//camera_changed",
        [this](const GraphEvent::Info&, const GraphEvent::Data&) { camera_moved = true; });
    io_layout.register_event_listener("//geometry_changed,//transform_changed",
                                      [this](const GraphEvent::Info&, const GraphEvent::Data&) {
                                          pixel_stats_dirty = true;
                                          maps_dirty = true;
                                          kept_dirty = true;
                                          heat_dirty = true;
                                          // a selection on its pixel sees what moved there
                                          if (!reproject && capture != CAPTURE_FROZEN) {
                                              repick();
                                          }
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
    ensure_reference(io.get_allocator(), submission);

    const PathRecordCapacities capacities = path_record_capacities(io[con_records]->get_size());
    const uint32_t observed_paths = apply_readback_feedback(info, capacities);
    const bool ready = ready_camera(io) != nullptr;

    record_clears(io, info, submission);
    const FrameActivity activity = update_frame(io, ready);
    params = build_params(io, info, activity, observed_paths);
    bind_globals(io);

    if (activity.active) {
        record_analysis(io, info, submission, capacities, activity);
    }
    record_compose(io, info, submission);
    if (activity.active) {
        record_map_panels(io, info, submission);
    }
    // an export takes the lines and labels with it, and the window shows it that frame
    if (export_next) {
        if (activity.active) {
            record_overlay(io, info, submission, false);
        }
        record_labels(io, info, submission, false);
    }
    // the window gets the lines at its own resolution, out only where something reads it
    const bool presented = record_present(io, submission);
    if (!export_next) {
        const bool labels_into_out = !io.is_connected(con_canvas);
        if (presented && activity.active) {
            record_overlay(io, info, submission, true);
        }
        if (presented && !labels_into_out) {
            record_labels(io, info, submission, true);
        }
        if (activity.active && (!presented || io.is_connected(con_out))) {
            record_overlay(io, info, submission, false);
        }
        if (labels_into_out) {
            record_labels(io, info, submission, false);
        }
    }
    record_export(io, info, submission);
    if (ready) {
        record_readback(io, info, submission);
    }
    return std::exchange(pending_reconnect, false) ? NodeStatusFlags{NEEDS_RECONNECT}
                                                   : NodeStatusFlags{};
}

void PathDebugNode::handle_input(const NodeIO& io) {
    if (io.is_connected(con_controller)) {
        const InputControllerHandle& controller = io[con_controller];
        if (controller && controller != registered_controller.lock()) {
            controller->add_listener(input, PathDebugInput::PRIORITY);
            controller->add_listener(input->observer(), PathDebugInput::OBSERVER_PRIORITY);
            registered_controller = controller;
        }
    }
    input->set_modifier(MODIFIER_KEYS.at(modifier));
    const PathDebugInput::Events events = input->consume_events();
    if (events.clear) {
        clear_selection();
    }
    if (events.panel_scroll != 0.) {
        panel_scroll = std::max(
            0, panel_scroll - static_cast<int32_t>(events.panel_scroll * PANEL_SCROLL_PIXELS));
    }
    if (events.radius_scroll != 0. && selected_pos_valid && select_kind == SELECT_REGION) {
        selected_radius =
            std::max(MIN_REGION_RADIUS,
                     selected_radius *
                         std::pow(REGION_SCROLL_FACTOR, static_cast<float>(events.radius_scroll)));
        invalidate_selection();
    }
    if (const std::optional<PathDebugInput::Click>& clicked = events.click) {
        const auto hit = presenting ? panel_at(clicked->pos * present_density) : std::nullopt;
        if (hit) {
            if (hit->on_close) {
                panel_enabled[hit->channel] = false;
            }
        } else if (const auto click = clicked->select ? to_image(clicked->pos) : std::nullopt) {
            select_at(*click);
        }
    }

    cursor_pixel.reset();
    probe_panel.reset();
    probe_at.reset();
    if (const auto cursor = input->cursor(); cursor && presenting) {
        probe_at = *cursor * present_density;
        if (const auto hit = panel_at(*probe_at)) {
            if (!hit->on_close) {
                probe_panel = hit;
            }
        } else {
            cursor_pixel = to_image(*cursor);
        }
    }
}

void PathDebugNode::select_at(const int2& pixel) {
    selected_pixel = pixel;
    selected_in_records = pixel;
    has_selection = true;
    reset_selected_point();
    point_behind_surface = false;
    request_pick(pixel);
    invalidate_selection();
}

void PathDebugNode::set_select_kind(const int32_t kind) {
    select_kind = kind;
    // a region resolves its pixel through clear glass, a point does not
    if (has_selection) {
        select_at(selected_pixel);
    }
}

void PathDebugNode::set_region_match(const int32_t match) {
    region_match = match;
    invalidate_selection();
}

std::optional<float2> PathDebugNode::project_point(Camera& camera) const {
    const float4 clip = mul(camera.get_view_projection_matrix(), float4(selected_pos, 1.f));
    if (clip.w <= camera.get_near_plane()) {
        return std::nullopt;
    }
    const float2 ndc = float2(clip.x, clip.y) / clip.w;
    return float2(0.5f + (0.5f * ndc.x), 0.5f - (0.5f * ndc.y)) *
           float2(static_cast<float>(extent.width), static_cast<float>(extent.height));
}

void PathDebugNode::reproject_pick(const NodeIO& io) {
    const CameraHandle camera = ready_camera(io);
    if (!reproject || !has_selection || !selected_pos_valid || !camera) {
        selected_offscreen = false;
        return;
    }
    const std::optional<float2> projected = project_point(*camera);
    const float2 at = projected.value_or(float2(-1.f)) + selected_pixel_offset;
    selected_offscreen = !projected || at.x < 0.f || at.y < 0.f ||
                         at.x >= static_cast<float>(extent.width) ||
                         at.y >= static_cast<float>(extent.height);
    if (!selected_offscreen) {
        selected_pixel = int2(floor(at));
    }
}

uint32_t PathDebugNode::apply_readback_feedback(const NodeProcessInfo& info,
                                                const PathRecordCapacities& capacities) {
    if (info.get_iteration() == 0) {
        readback_valid = false;
        if (auto_keep_prob) {
            keep_prob = MIN_KEEP_PROB;
        }
    }
    const auto tick = [](bool& drawn, uint32_t& frames) {
        frames = std::exchange(drawn, false) ? 0 : std::min(frames + 1, UI_IDLE_FRAMES);
    };
    tick(window_drawn, frames_since_window);
    tick(drawn_paths_shown, frames_since_drawn_paths);
    tick(pixel_section_shown, frames_since_pixel_section);
    tick(path_statistics_shown, frames_since_path_statistics);

    if (!readback_valid) {
        return capacities.path_capacity;
    }
    if (latest_readback.stats[PATH_DEBUG_STATS_COUNT] > MAX_MAP_COUNT) {
        maps_dirty = true;
    }
    adapt_keep_prob(capacities);
    if (const PointPaths& point = latest_readback.point; point.at_point + point.in_front > 0) {
        point_behind_surface = point.at_point == 0;
    }
    return std::min(latest_readback.frame.paths, capacities.path_capacity);
}

void PathDebugNode::adapt_keep_prob(const PathRecordCapacities& capacities) {
    if (!auto_keep_prob || capture != CAPTURE_LIVE || selects_point()) {
        return;
    }
    const float vertex_fill = static_cast<float>(latest_readback.frame.vertices) /
                              static_cast<float>(capacities.vertex_capacity);
    const float path_fill = static_cast<float>(latest_readback.frame.paths) /
                            static_cast<float>(capacities.path_capacity);
    const float fill = std::max(vertex_fill, path_fill);
    const float keep_used = std::bit_cast<float>(latest_readback.frame.keep_prob);
    if (keep_used > 0.f && (fill > 1.f || fill < MIN_FILL)) {
        keep_prob = std::clamp(keep_used * TARGET_FILL / std::max(fill, 1e-3f), MIN_KEEP_PROB, 1.f);
    }
}

void PathDebugNode::resolve_pick(Camera& camera) {
    const Pick& pick = latest_readback.pick;
    // the records know the first vertex and what lies behind clear glass while they were traced
    // from this view; otherwise the gbuffer answers for the view as it is now
    const bool records_cover_pick =
        traced_pixel_filter == PATH_RECORD_ALL_PIXELS ||
        traced_pixel_filter == path_record_pack_pixel(uint2(pending_pick_pixel));
    const bool records_answer = !gbuffer_connected || capture == CAPTURE_LIVE ||
                                (capture == CAPTURE_FROZEN && !camera_moved && records_cover_pick);
    if (!pending_pick || !readback_valid ||
        (records_answer ? pick.record_seq : pick.surface_seq) != pending_pick_seq) {
        return;
    }
    pending_pick = false;
    const float3 surface = pick.surface;
    const float3 hit = !records_answer                ? surface
                       : select_kind == SELECT_REGION ? pick.seen
                                                      : pick.first;
    const float distance_to_camera = distance(camera.get_position(), hit);
    // the sky keeps the previous surface, or selects nothing
    if (distance_to_camera >= PATH_DEBUG_ENV_DISTANCE * 0.5f) {
        if (!selected_pos_valid) {
            clear_selection();
        }
        return;
    }
    selected_pos = hit;
    // the gbuffer's normal only where it shows that surface
    const float3 normal = pick.normal;
    selected_normal = gbuffer_connected && pick.surface_seq == pending_pick_seq &&
                              length(normal) > 0.f &&
                              distance(surface, hit) <= 0.02f * distance_to_camera
                          ? normalize(normal)
                          : float3(0.f);
    if (selected_radius <= 0.f) {
        selected_radius = REGION_RADIUS_PIXELS * pixel_angle(camera) * distance_to_camera;
    }
    selected_pos_valid = true;
    selected_pixel_offset = float2(0.f);
    if (const std::optional<float2> projected = project_point(camera)) {
        selected_pixel_offset = float2(pending_pick_pixel) + 0.5f - *projected;
    }
    invalidate_selection();
}

CameraHandle PathDebugNode::ready_camera(const NodeIO& io) const {
    const SceneHandle& scene = io[con_scene];
    return scene && scene->is_ready() ? scene->get_active_camera() : CameraHandle();
}

float PathDebugNode::pixel_angle(const Camera& camera) const {
    return 2.f * std::tan(camera.get_field_of_view_vertical() * 0.5f) /
           static_cast<float>(extent.height);
}

void PathDebugNode::record_clears(const NodeIO& io,
                                  const NodeProcessInfo& info,
                                  Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    const bool first_frame = info.get_iteration() == 0;
    cmd->barrier({compute_to_transfer(state_buffer), compute_to_transfer(matches_buffer),
                  compute_to_transfer(filtered_buffer), compute_to_transfer(stats_buffer),
                  compute_to_transfer(maps_buffer), compute_to_transfer(grid_buffer),
                  compute_to_transfer(moments_buffer), compute_to_transfer(kept_buffer)});
    // held records stay with the view they were traced from
    if (camera_moved && capture != CAPTURE_FROZEN && !was_frozen) {
        camera_moved = false;
        pixel_stats_dirty = true;
        maps_dirty = true;
        kept_dirty |= !keep_on_camera_move;
        // a selection on its pixel now sees another surface
        if (!reproject) {
            repick();
        }
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
        check_draws = 0;
        pdf_stale = true;
    }
    if (heat_dirty || first_frame) {
        cmd->fill(grid_buffer);
        heat_dirty = false;
        heat_splatted = false;
    }
    // a reset waits for something to read the moments or the re-render
    const bool pixel_buffers_read =
        pixel_stats_read() || backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_FILTERED) ||
        io.is_connected(con_filtered);
    if ((pixel_stats_dirty && pixel_buffers_read) || first_frame) {
        cmd->fill(moments_buffer);
        cmd->fill(filtered_buffer);
        pixel_stats_dirty = false;
        pixel_stats_accumulated = false;
        pixel_stats_generation++;
        pixel_history.clear();
    }
    if (kept_dirty || first_frame) {
        cmd->fill(kept_buffer);
        kept_dirty = false;
        kept_updated = false;
    }
    cmd->barrier({transfer_to_compute(state_buffer), transfer_to_compute(stats_buffer),
                  transfer_to_compute(maps_buffer), transfer_to_compute(grid_buffer),
                  transfer_to_compute(moments_buffer), transfer_to_compute(filtered_buffer),
                  transfer_to_compute(matches_buffer), transfer_to_compute(kept_buffer)});
}

PathDebugNode::FrameActivity PathDebugNode::update_frame(const NodeIO& io, const bool ready) {
    // this frame's records were traced with the last frame's pixel filter, unless it held them
    if (params.freeze == 0) {
        traced_pixel_filter = params.record_pixel;
    }
    if (const CameraHandle camera = ready_camera(io)) {
        resolve_pick(*camera);
    }
    reproject_pick(io);
    selected_pixel = clamp(selected_pixel, int2(0), to_int2(extent) - 1);
    // held records keep matching the pixel they were traced through
    const bool held =
        capture == CAPTURE_FROZEN && readback_valid && latest_readback.frame.paths > 0;
    if (!has_selection) {
        selected_in_records.reset();
    } else if (!held && traced_pixel_filter != PATH_RECORD_ALL_PIXELS) {
        // one pixel's records follow the point a frame behind
        selected_in_records =
            traced_pixel_filter != PATH_DEBUG_RECORD_NO_PIXEL
                ? std::optional<int2>(int2(path_record_unpack_pixel(traced_pixel_filter)))
                : std::optional<int2>();
    } else if (!held) {
        selected_in_records =
            selection_visible() ? std::optional<int2>(selected_pixel) : std::optional<int2>();
    }
    reference_input_connected = io.is_connected(con_reference);
    backdrop_missing = (shows_reference() && !has_reference()) ||
                       (backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_INPUT) &&
                        !io.is_connected(con_backdrop));

    FrameActivity activity{};
    activity.active = ready && capture != CAPTURE_OFF;
    activity.frozen =
        std::exchange(was_frozen, capture == CAPTURE_FROZEN) || capture == CAPTURE_FROZEN;
    // an isolated path keeps its slot
    activity.keep_update = (!activity.frozen && !isolate_path) || !kept_updated;
    kept_updated |= activity.keep_update && activity.active;
    activity.pixel_stats_accumulate =
        pixel_stats_read() && (!activity.frozen || !pixel_stats_accumulated);
    pixel_stats_accumulated |= activity.pixel_stats_accumulate && activity.active;
    activity.heat_splat = !activity.frozen || !heat_splatted;

    activity.panel_mask = panels_enabled && activity.active ? shown_panel_mask() : 0;
    update_panel_layout(io, activity.panel_mask);
    activity.sphere_channel =
        (sphere_channel < 0 || !gbuffer_connected || capture == CAPTURE_OFF || !maps_at_point())
            ? PATH_DEBUG_MAP_NONE
            : static_cast<uint32_t>(sphere_channel);
    activity.maps_active = activity.panel_mask != 0 ||
                           activity.sphere_channel != PATH_DEBUG_MAP_NONE ||
                           frames_since_path_statistics < UI_IDLE_FRAMES;
    activity.maps_accumulate = activity.maps_active && (!activity.frozen || map_frames == 0);
    activity.batch_end =
        activity.maps_accumulate &&
        (map_frames + 1) % static_cast<uint32_t>(std::max(map_batch_frames, 1)) == 0;
    activity.run_check = std::exchange(check_requested, false);
    activity.check_shown = (activity.panel_mask & (1u << PATH_DEBUG_MAP_CHECK)) != 0 ||
                           activity.sphere_channel == PATH_DEBUG_MAP_CHECK;
    return activity;
}

void PathDebugNode::update_panel_layout(const NodeIO& io, const uint32_t panel_mask) {
    const bool canvas_connected = io.is_connected(con_canvas);
    if (const ImageHandle canvas = canvas_connected ? io[con_canvas].get_image() : ImageHandle()) {
        label_area = to_int2(canvas->get_extent());
        label_scale = io.is_connected(con_window) && io[con_window]
                          ? io[con_window]->get_display_scale()
                          : 1.f;
    } else if (!canvas_connected) {
        label_area = to_int2(out_extent);
        label_scale = 1.f;
    }
    const int32_t panel_limit =
        std::min(label_area.x, label_area.y) - (2 * label_px(PATH_DEBUG_PANEL_MARGIN));
    panel_px = std::clamp(label_px(static_cast<uint32_t>(std::max(panel_size, 0))), MIN_PANEL_PX,
                          std::max(panel_limit, MIN_PANEL_PX));
    const auto panel_count = static_cast<int32_t>(std::popcount(panel_mask));
    const int32_t column = (panel_count * (panel_px + label_px(PATH_DEBUG_PANEL_GAP))) +
                           label_px(PATH_DEBUG_PANEL_MARGIN);
    panel_overflow = std::max(0, column - label_area.y);
    panel_scroll = std::clamp(panel_scroll, 0, panel_overflow);
    if (panel_mask != 0 && presenting) {
        input->set_panel_area(
            float2(static_cast<float>(panel_left()), 0.f) / present_density,
            float2(static_cast<float>(label_area.x),
                   static_cast<float>(std::min(column - panel_scroll, label_area.y))) /
                present_density);
    } else {
        input->set_panel_area(float2(0.f), float2(0.f));
    }
}

PathDebugParams PathDebugNode::build_params(const NodeIO& io,
                                            const NodeProcessInfo& info,
                                            const FrameActivity& activity,
                                            const uint32_t observed_paths) const {
    PathDebugParams p{};
    p.dim_x = extent.width;
    p.dim_y = extent.height;
    p.out_x = out_extent.width;
    p.out_y = out_extent.height;
    build_selection_params(p, io);
    build_capture_params(p);
    build_overlay_params(p, activity);
    build_map_params(p, activity, observed_paths);
    build_heat_params(p, activity);
    build_pixel_stats_params(p, io, activity);
    build_backdrop_params(p, info);
    if (capture == CAPTURE_OFF) {
        p.backdrop = PATH_DEBUG_BACKDROP_RENDER;
        p.heat_mode = PATH_DEBUG_HEAT_OFF;
        p.pixel_stats_view = PATH_DEBUG_PIXEL_STAT_OFF;
        p.filtered_active = 0;
        p.filtered_out = 0;
    }
    p.cursor_x = cursor_pixel ? static_cast<uint32_t>(cursor_pixel->x) : PATH_DEBUG_NO_PIXEL;
    p.cursor_y = cursor_pixel ? static_cast<uint32_t>(cursor_pixel->y) : PATH_DEBUG_NO_PIXEL;
    p.probe_channel = probe_panel ? probe_panel->channel : PATH_DEBUG_MAP_NONE;
    p.probe_bin = probe_panel ? probe_panel->bin : 0;
    return p;
}

void PathDebugNode::build_selection_params(PathDebugParams& p, const NodeIO& io) const {
    p.sel_pixel_x =
        selection_visible() ? static_cast<uint32_t>(selected_pixel.x) : PATH_DEBUG_NO_PIXEL;
    p.sel_pixel_y =
        selection_visible() ? static_cast<uint32_t>(selected_pixel.y) : PATH_DEBUG_NO_PIXEL;
    p.sel_record_x = !has_selection        ? PATH_DEBUG_NO_PIXEL
                     : selected_in_records ? static_cast<uint32_t>(selected_in_records->x)
                                           : PATH_DEBUG_OFFSCREEN;
    p.sel_record_y = !has_selection        ? PATH_DEBUG_NO_PIXEL
                     : selected_in_records ? static_cast<uint32_t>(selected_in_records->y)
                                           : PATH_DEBUG_OFFSCREEN;
    p.sel_point_x = selected_pos.x;
    p.sel_point_y = selected_pos.y;
    p.sel_point_z = selected_pos.z;
    p.sel_normal_x = selected_normal.x;
    p.sel_normal_y = selected_normal.y;
    p.sel_normal_z = selected_normal.z;
    p.sel_region_radius = selected_pos_valid ? selected_radius : 0.f;
    p.sel_point_valid = has_selection && selected_pos_valid ? 1 : 0;
    p.sel_mode = selection_mode();
    // jitter, the pixel grid and a grazing surface move the hit within a few pixel footprints
    p.sel_point_tolerance = 0.f;
    if (const CameraHandle camera = ready_camera(io); reproject && selected_pos_valid && camera) {
        p.sel_point_tolerance =
            (0.05f + (4.f * pixel_angle(*camera))) * distance(camera->get_position(), selected_pos);
    }
    p.sel_scatters_min = static_cast<uint32_t>(scatter_range.x);
    p.sel_scatters_max = scatter_range.y >= SCATTER_UNLIMITED
                             ? PATH_DEBUG_NO_SCATTER_LIMIT
                             : static_cast<uint32_t>(scatter_range.y);
    p.sel_method_mask = method_mask;
    p.sel_method_mode = static_cast<uint32_t>(method_mode);
    p.sel_finite = static_cast<uint32_t>(contribution_filter);
    p.sel_material =
        material_id >= 0 ? static_cast<uint32_t>(material_id) : PATH_RECORD_MATERIAL_NONE;
    p.expr_0 = compiled_expression.words[0];
    p.expr_1 = compiled_expression.words[1];
    p.expr_2 = compiled_expression.words[2];
    p.expr_3 = compiled_expression.words[3];
    p.expr_count =
        compiled_expression.valid
            ? compiled_expression.token_count | (compiled_expression.nee ? PATH_DEBUG_EXPR_NEE : 0u)
            : 0;
    p.pick_x = pending_pick ? static_cast<uint32_t>(pending_pick_pixel.x) : PATH_DEBUG_NO_PIXEL;
    p.pick_y = pending_pick ? static_cast<uint32_t>(pending_pick_pixel.y) : PATH_DEBUG_NO_PIXEL;
    p.pick_seq = pending_pick_seq;
}

void PathDebugNode::build_capture_params(PathDebugParams& p) const {
    const bool one_pixel = selects_point();
    p.freeze = capture == CAPTURE_FROZEN ? 1 : 0;
    p.capture_off = capture == CAPTURE_OFF ? 1 : 0;
    p.keep_prob = one_pixel ? 1.f : keep_prob;
    // a pending pick is answered from the records of its pixel
    p.record_pixel = !one_pixel            ? PATH_RECORD_ALL_PIXELS
                     : pending_pick        ? path_record_pack_pixel(uint2(pending_pick_pixel))
                     : selection_visible() ? path_record_pack_pixel(uint2(selected_pixel))
                                           : PATH_DEBUG_RECORD_NO_PIXEL;
}

void PathDebugNode::build_overlay_params(PathDebugParams& p, const FrameActivity& activity) const {
    p.overlay_rank = static_cast<uint32_t>(overlay_rank);
    p.overlay_max_paths = static_cast<uint32_t>(overlay_max_paths);
    p.overlay_keep_update = activity.keep_update ? 1 : 0;
    p.overlay_brightest_share = overlay_brightest_share;
    p.overlay_key_decay =
        overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_RESAMPLED) && resample_history > 0.f
            ? std::exp(-1.f / resample_history)
            : 1.f;
    p.overlay_alpha = overlay_opacity;
    p.overlay_thickness = overlay_thickness;
    p.overlay_color_scale = overlay_exposure;
    p.overlay_depth_mode = static_cast<uint32_t>(overlay_depth_mode);
    p.overlay_color = static_cast<uint32_t>(overlay_color);
    p.overlay_isolate = isolate_path.value_or(PATH_DEBUG_NO_PATH);
}

void PathDebugNode::build_map_params(PathDebugParams& p,
                                     const FrameActivity& activity,
                                     const uint32_t observed_paths) const {
    p.map_res = allocated_map_res;
    p.map_transform = static_cast<uint32_t>(map_transform);
    p.map_frame = gbuffer_connected && selection_visible() ? static_cast<uint32_t>(map_frame)
                                                           : PATH_DEBUG_FRAME_WORLD;
    p.map_pdf_ref = static_cast<uint32_t>(map_pdf_ref);
    p.map_draws = static_cast<uint32_t>(map_draws);
    p.map_batch_end = activity.batch_end ? 1 : 0;
    p.map_panel_mask = activity.panel_mask;
    p.map_exposure = map_exposure;
    p.want_top_list = frames_since_drawn_paths < UI_IDLE_FRAMES ? 1 : 0;
    p.stats_stride = (analysis_paths <= 0 || selects_point())
                         ? 1
                         : std::max(observed_paths / static_cast<uint32_t>(analysis_paths), 1u);
    p.sphere_channel = activity.sphere_channel;
    p.sphere_radius = sphere_size;
    p.wi_mode = static_cast<uint32_t>(wi_mode);
    p.wi_theta = radians(wi_theta_deg);
    p.wi_phi = radians(wi_phi_deg);
}

void PathDebugNode::build_heat_params(PathDebugParams& p, const FrameActivity& activity) const {
    p.heat_mode = gbuffer_connected && false_color == FALSE_COLOR_HEAT
                      ? PATH_DEBUG_HEAT_DENSITY + static_cast<uint32_t>(heat_quantity)
                      : PATH_DEBUG_HEAT_OFF;
    p.heat_bounce_min = static_cast<uint32_t>(heat_bounce_range.x);
    p.heat_bounce_max = static_cast<uint32_t>(heat_bounce_range.y);
    p.heat_decay =
        activity.heat_splat && heat_history > 0.f ? 1.f - (1.f / std::max(heat_history, 1.f)) : 1.f;
    p.heat_cell_size = heat_cell_size;
    p.heat_exposure = heat_exposure;
    p.heat_smooth = heat_smooth ? 1 : 0;
    p.false_color_opacity = false_color_opacity;
    p.grid_slots =
        static_cast<uint32_t>(grid_buffer->get_size() / sizeof(uint32_t) / PATH_DEBUG_GRID_REGIONS);
}

void PathDebugNode::build_pixel_stats_params(PathDebugParams& p,
                                             const NodeIO& io,
                                             const FrameActivity& activity) const {
    p.pixel_stats_accumulate = activity.pixel_stats_accumulate ? 1 : 0;
    p.pixel_stats_view = false_color == FALSE_COLOR_PIXEL_STAT
                             ? PATH_DEBUG_PIXEL_STAT_MEAN + static_cast<uint32_t>(pixel_statistic)
                             : PATH_DEBUG_PIXEL_STAT_OFF;
    p.pixel_stats_scale = pixel_stats_exposure;
    p.filtered_out = io.is_connected(con_filtered) ? 1 : 0;
    p.filtered_active =
        backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_FILTERED) || p.filtered_out != 0 ? 1
                                                                                              : 0;
    p.filtered_exposure = filtered_exposure;
}

void PathDebugNode::build_backdrop_params(PathDebugParams& p, const NodeProcessInfo& info) const {
    p.backdrop_split = backdrop_split;
    p.difference_gain = difference_gain;
    p.backdrop = backdrop_missing ? PATH_DEBUG_BACKDROP_RENDER : static_cast<uint32_t>(backdrop);
    if (reference_flip && (p.backdrop == PATH_DEBUG_BACKDROP_REFERENCE ||
                           p.backdrop == PATH_DEBUG_BACKDROP_SPLIT_REFERENCE)) {
        const auto phase = static_cast<uint64_t>(info.get_elapsed() * reference_flip_hz);
        p.backdrop = (phase & 1u) == 0 ? PATH_DEBUG_BACKDROP_RENDER : PATH_DEBUG_BACKDROP_REFERENCE;
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
        {"kept", kept_buffer},
        {"in_records", io[con_records]},
    }};
    for (const auto& [name, buffer] : buffers) {
        if (auto c = cursor.find(name); c.is_valid()) {
            c = buffer;
        }
    }
    if (auto c = cursor.find("out_panels"); c.is_valid() && panel_texture) {
        c = panel_texture->get_view();
    }
    if (auto c = cursor.find("in_reference"); c.is_valid()) {
        if (io.is_connected(con_reference)) {
            c = io[con_reference].get_texture();
        } else if (reference_texture) {
            c = reference_texture;
        }
    }
    if (auto c = cursor.find("in_backdrop"); c.is_valid()) {
        c = io.is_connected(con_backdrop) ? io[con_backdrop].get_texture()
                                          : io[con_src].get_texture();
    }
    if (auto c = cursor.find("params"); c.is_valid()) {
        c = params;
    }
    if (auto c = cursor.find("camera"); c.is_valid()) {
        if (const CameraHandle camera = ready_camera(io)) {
            camera->write_to(c);
        }
    }
    if (auto c = cursor.find("gbuffer"); c.is_valid() && gbuffer_connected) {
        c = io[con_gbuffer].r();
    }
}

std::array<std::optional<ComputeKernel>*, 12> PathDebugNode::kernels() {
    return {&select_kernel,     &threshold_kernel,  &collect_kernel,       &keep_kernel,
            &stats_kernel,      &map_reduce_kernel, &map_view_kernel,      &compose_kernel,
            &heat_decay_kernel, &heat_splat_kernel, &filter_render_kernel, &finalize_kernel};
}

void PathDebugNode::bind_globals(const NodeIO& io) {
    for (std::optional<ComputeKernel>* const kernel : kernels()) {
        bind_node_buffers((*kernel)->globals_cursor(), io);
    }
}

PipelineHandle PathDebugNode::bind_scene_pipeline(ScenePipeline& kernel,
                                                  const NodeIO& io,
                                                  const NodeProcessInfo& info,
                                                  Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    const SceneHandle& scene = io[con_scene];
    const auto ep = kernel.entry_point.get();
    const auto pipe = kernel.pipeline.get();
    const auto globals = kernel.globals.get();
    bind_node_buffers(globals->get_cursor(), io);
    cmd->bind(pipe);
    ep->bind("scene", scene->get_shader_object(), cmd, pipe, info.get_shader_object_allocator());
    ep->bind_global(globals, cmd, pipe, info.get_shader_object_allocator());
    return pipe;
}

void PathDebugNode::record_analysis(const NodeIO& io,
                                    const NodeProcessInfo& info,
                                    Submission& submission,
                                    const PathRecordCapacities& capacities,
                                    const FrameActivity& activity) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    const SceneHandle& scene = io[con_scene];
    const uint32_t path_groups = groups(capacities.path_capacity);
    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "select");
        select_kernel->bind(io, info, submission);
        cmd->dispatch(path_groups, 1, 1);
    }
    cmd->barrier(compute_to_compute(matches_buffer));
    if (params.filtered_active != 0) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "filter render");
        filter_render_kernel->bind(io, info, submission);
        cmd->dispatch(path_groups, 1, 1);
    }
    if (activity.maps_accumulate) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "stats");
        const uint32_t stats_threads = ceil_div(capacities.path_capacity, params.stats_stride);
        stats_kernel->bind(io, info, submission);
        cmd->dispatch(groups(stats_threads), 1, 1);
        map_frames++;
    }
    if (pdf_stale && activity.maps_active && gbuffer_connected) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "bsdf map");
        ensure_scene_pipeline(io.get_context(), io.get_allocator(), bsdf_map, scene,
                              BSDF_MAP_MODULE);
        bind_scene_pipeline(bsdf_map, io, info, submission);
        cmd->dispatch(vk::Extent3D{params.map_res, params.map_res, 1}, PATH_DEBUG_WG_2D,
                      PATH_DEBUG_WG_2D);
        pdf_stale = false;
    }
    // a check tops the draws up and integrates anew, a shown BSDF sample() draws up to that
    const uint32_t check_missing =
        PATH_DEBUG_CHECK_TARGET_DRAWS - std::min(check_draws, PATH_DEBUG_CHECK_TARGET_DRAWS);
    const uint32_t check_new_draws = activity.run_check ? check_missing
                                     : activity.check_shown
                                         ? std::min(PATH_DEBUG_CHECK_FRAME_DRAWS, check_missing)
                                         : 0;
    if ((activity.run_check || check_new_draws > 0) && gbuffer_connected) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "bsdf check");
        const PathDebugCheckPass pass{
            ++check_seq,
            ceil_div(check_new_draws, PATH_DEBUG_CHECK_SAMPLES_PER_CELL),
            activity.run_check ? 1u : 0u,
        };
        if (activity.run_check) {
            cmd->barrier({compute_to_transfer(stats_buffer), compute_to_compute(maps_buffer)});
            cmd->fill(stats_buffer, sizeof(uint32_t) * PATH_DEBUG_CHECK_QUADRATURE,
                      sizeof(uint32_t) * PATH_DEBUG_CHECK_QUADRATURE_UINTS);
            cmd->barrier(transfer_to_compute(stats_buffer));
        } else {
            cmd->barrier({compute_to_compute(stats_buffer), compute_to_compute(maps_buffer)});
        }
        ensure_scene_pipeline(io.get_context(), io.get_allocator(), bsdf_check, scene,
                              BSDF_CHECK_MODULE);
        const uint32_t rows = activity.run_check
                                  ? PATH_DEBUG_CHECK_DIM
                                  : ceil_div(pass.sample_threads, PATH_DEBUG_CHECK_DIM);
        const PipelineHandle pipe = bind_scene_pipeline(bsdf_check, io, info, submission);
        cmd->push_constant(pipe, pass);
        cmd->dispatch(vk::Extent3D{PATH_DEBUG_CHECK_DIM, rows, 1}, PATH_DEBUG_WG_2D,
                      PATH_DEBUG_WG_2D);
        check_draws += pass.sample_threads * PATH_DEBUG_CHECK_SAMPLES_PER_CELL;
    }
    cmd->barrier({compute_to_compute(state_buffer), compute_to_compute(stats_buffer),
                  compute_to_compute(maps_buffer)});
    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "reduce+threshold");
        if (activity.maps_active) {
            map_reduce_kernel->bind(io, info, submission);
            cmd->dispatch(groups(params.map_res * params.map_res), 1, 1);
            cmd->barrier(compute_to_compute(stats_buffer));
        }
        threshold_kernel->bind(io, info, submission);
        cmd->dispatch(1, 1, 1);
    }
    cmd->barrier({compute_to_compute(state_buffer), compute_to_compute(stats_buffer)});
    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "collect");
        collect_kernel->bind(io, info, submission);
        cmd->dispatch(path_groups, 1, 1);
    }
    if (params.overlay_keep_update != 0 && keeps_top()) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "keep");
        cmd->barrier({compute_to_compute(state_buffer), compute_to_compute(kept_buffer)});
        keep_kernel->bind(io, info, submission);
        cmd->dispatch(1, 1, 1);
    }
    if (params.heat_mode != PATH_DEBUG_HEAT_OFF) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "heat");
        cmd->barrier(compute_to_compute(state_buffer));
        heat_decay_kernel->bind(io, info, submission);
        cmd->dispatch(groups(params.grid_slots), 1, 1);
        if (activity.heat_splat) {
            cmd->barrier(compute_to_compute(grid_buffer));
            heat_splat_kernel->bind(io, info, submission);
            cmd->dispatch(path_groups, 1, 1);
            heat_splatted = true;
        }
    }
    cmd->barrier({compute_to_compute(stats_buffer), compute_to_compute(maps_buffer),
                  compute_to_compute(grid_buffer), compute_to_compute(state_buffer),
                  compute_to_compute(moments_buffer), compute_to_compute(filtered_buffer),
                  compute_to_compute(kept_buffer)});
}

void PathDebugNode::record_compose(const NodeIO& io,
                                   const NodeProcessInfo& info,
                                   Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "compose");
    compose_kernel->bind(io, info, submission);
    cmd->dispatch(vk::Extent3D{std::max(extent.width, out_extent.width),
                               std::max(extent.height, out_extent.height), 1},
                  PATH_DEBUG_WG_2D, PATH_DEBUG_WG_2D);
}

void PathDebugNode::record_overlay(const NodeIO& io,
                                   const NodeProcessInfo& info,
                                   Submission& submission,
                                   const bool onto_canvas) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    const BufferHandle& records = io[con_records];
    MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, onto_canvas ? "lines (window)" : "lines");
    const TextureHandle& target =
        onto_canvas ? io[con_canvas].get_texture() : io[con_out].get_texture(0);
    const ImageHandle& image = target->get_image();
    OverlayTarget& overlay = onto_canvas ? overlay_canvas : overlay_out;
    ensure_overlay_target(io.get_context(), io.get_allocator(), overlay, image->get_format());

    const vk::Offset3D lower = onto_canvas ? present_lower : vk::Offset3D{};
    const vk::Offset3D upper = onto_canvas ? present_upper : to_offset(out_extent);
    const vk::Rect2D view{{lower.x, lower.y},
                          {static_cast<uint32_t>(std::max(upper.x - lower.x, 0)),
                           static_cast<uint32_t>(std::max(upper.y - lower.y, 0))}};
    if (view.extent.width == 0 || view.extent.height == 0) {
        return;
    }
    const PathDebugOverlayView overlay_view{
        static_cast<float>(view.offset.x),     static_cast<float>(view.offset.y),
        static_cast<float>(view.extent.width), static_cast<float>(view.extent.height),
        onto_canvas ? label_scale : 1.f,
    };

    const auto vertex = overlay_vertex.get();
    const auto pipe = overlay.pipeline.get();
    const auto globals = overlay.globals.get();
    bind_node_buffers(globals->get_cursor(), io);

    cmd->barrier({compute_to_vertex(state_buffer), compute_to_vertex(kept_buffer),
                  compute_to_vertex(records)});
    cmd->barrier(to_color_attachment(image));
    const vk::Extent3D target_extent = image->get_extent();
    const vk::RenderingAttachmentInfo color{*target->get_view(),
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
    cmd->push_constant(pipe, overlay_view);

    cmd->begin_rendering(vk::RenderingInfo{
        {}, vk::Rect2D{{0, 0}, {target_extent.width, target_extent.height}}, 1, 0, color});
    cmd->set_viewport(
        vk::Viewport{overlay_view.x, overlay_view.y, overlay_view.w, overlay_view.h, 0.f, 1.f});
    cmd->set_scissor(view);
    cmd->draw(6, path_debug_overlay_instances(params.overlay_max_paths));
    cmd->end_rendering();
    if (!onto_canvas) {
        cmd->barrier(from_color_attachment(image));
    }
}

void PathDebugNode::record_map_panels(const NodeIO& io,
                                      const NodeProcessInfo& info,
                                      Submission& submission) {
    if (params.map_panel_mask == 0) {
        return;
    }
    const CommandBufferHandle& cmd = submission.get_cmd();
    MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "map panels");
    const ImageHandle& atlas = panel_texture->get_image();
    cmd->barrier(atlas->barrier2(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
                                 vk::AccessFlagBits2::eShaderWrite,
                                 vk::PipelineStageFlagBits2::eFragmentShader,
                                 vk::PipelineStageFlagBits2::eComputeShader));
    map_view_kernel->bind(io, info, submission);
    cmd->dispatch(vk::Extent3D{params.map_res, params.map_res, PATH_DEBUG_MAP_CHANNEL_COUNT},
                  PATH_DEBUG_WG_2D, PATH_DEBUG_WG_2D);
    cmd->barrier(atlas->barrier2(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
                                 vk::AccessFlagBits2::eShaderRead,
                                 vk::PipelineStageFlagBits2::eComputeShader,
                                 vk::PipelineStageFlagBits2::eFragmentShader));
}

void PathDebugNode::record_labels(const NodeIO& io,
                                  const NodeProcessInfo& info,
                                  Submission& submission,
                                  const bool onto_canvas) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "labels");
    const TextureHandle& target =
        onto_canvas ? io[con_canvas].get_texture() : io[con_out].get_texture(0);
    const WindowHandle window =
        onto_canvas && io.is_connected(con_window) ? io[con_window] : WindowHandle();
    if (!onto_canvas) {
        label_area = to_int2(out_extent);
        cmd->barrier(to_color_attachment(io[con_out]));
    }
    if (window && window != imgui_window.lock()) {
        imgui_window_backend = std::make_shared<ImGuiMerianWindowBackend>(
            imgui_ctx, window, PathDebugInput::UI_PRIORITY);
        imgui_window = window;
    }
    const auto delta_time = static_cast<float>(frametime.seconds());
    frametime.reset();
    if (window) {
        imgui_window_backend->new_frame(delta_time);
    } else {
        ImGuiIO& imgui_io = imgui_ctx->get_io();
        imgui_io.DisplaySize =
            ImVec2(static_cast<float>(label_area.x), static_cast<float>(label_area.y));
        imgui_io.DisplayFramebufferScale = ImVec2(1.f, 1.f);
        imgui_ctx->get_style().FontScaleDpi = label_scale;
        imgui_backend->new_frame(delta_time);
    }
    imgui_ctx->with_context([&] {
        if (window) {
            draw_window();
            draw_image_choice();
        }
        draw_panel_labels();
        draw_probe_tooltip();
    });
    imgui_renderer->render(cmd, target->get_view());
    if (!onto_canvas) {
        cmd->barrier(from_color_attachment(io[con_out]));
    }
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
    const auto format = IMAGE_EXPORT_FORMATS.at(export_format);
    image_export(io.get_allocator(), submission, info.get_profiler(), io[con_out],
                 vk::Extent2D{out_extent.width, out_extent.height},
                 std::filesystem::absolute(export_path + image_format_extension(format)), format,
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
    cmd->copy(
        stats_buffer, readback,
        vk::BufferCopy{0, offsetof(Readback, stats), sizeof(uint32_t) * PATH_DEBUG_STATS_UINTS});
    cmd->barrier({compute_to_compute(records), compute_to_compute(state_buffer),
                  stats_buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eTransfer,
                                                vk::PipelineStageFlagBits2::eComputeShader,
                                                vk::AccessFlagBits2::eTransferRead,
                                                vk::AccessFlagBits2::eShaderWrite)});
    finalize_kernel->bind(io, info, submission);
    cmd->dispatch(1, 1, 1);
    cmd->barrier(out_state_buffer->buffer_barrier2(
        vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eTransferRead));
    cmd->copy(out_state_buffer, readback,
              vk::BufferCopy{0, 0, sizeof(uint32_t) * PATH_DEBUG_OUT_UINTS});

    cmd->barrier(readback->buffer_barrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eHost,
        vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eHostRead));
    submission.sync_to_cpu([this, readback, recorded_pixel_stats = pixel_stats_generation]() {
        const Readback data = *readback->get_memory()->map_as<Readback>();
        readback->get_memory()->unmap();
        latest_readback = data;
        readback_valid = true;
        if (recorded_pixel_stats != pixel_stats_generation) {
            return;
        }
        pixel_history.push_back({std::bit_cast<float>(data.probe.selected_mean),
                                 std::bit_cast<float>(data.probe.selected_rel_error),
                                 std::bit_cast<float>(data.probe.selected_variance)});
        if (pixel_history.size() > PIXEL_HISTORY) {
            pixel_history.pop_front();
        }
    });
}

bool PathDebugNode::record_present(const NodeIO& io, Submission& submission) {
    if (!io.is_connected(con_canvas)) {
        presenting = false;
        return false;
    }
    // a frame without a canvas image keeps the last mapping, input still refers to it
    const ImageHandle& dst = io[con_canvas].get_image();
    if (!dst) {
        return false;
    }

    const CommandBufferHandle& cmd = submission.get_cmd();
    const ImageHandle& src = io[con_out].get_image();
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
    return true;
}

void PathDebugNode::ensure_reference(const ResourceAllocatorHandle& allocator,
                                     Submission& submission) {
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
    kept_dirty = true;
}

void PathDebugNode::repick() {
    if (!has_selection || pending_pick || selected_offscreen) {
        return;
    }
    request_pick(selected_pixel);
}

void PathDebugNode::request_pick(const int2& pixel) {
    pending_pick_pixel = pixel;
    pending_pick = true;
    pending_pick_seq++;
}

void PathDebugNode::reset_selected_point() {
    selected_offscreen = false;
    selected_pos_valid = false;
    selected_normal = float3(0.f);
    selected_radius = 0.f;
}

void PathDebugNode::clear_selection() {
    has_selection = false;
    reset_selected_point();
    pending_pick = false;
    isolate_path.reset();
    invalidate_selection();
}

void PathDebugNode::set_expression(const std::string& text) {
    expression = text;
    compiled_expression = compile_path_expression(text);
    invalidate_selection();
}

bool PathDebugNode::keeps_top() const {
    return path_debug_rank_keeps_top(static_cast<uint32_t>(overlay_rank));
}

bool PathDebugNode::pixel_stats_read() const {
    return false_color == FALSE_COLOR_PIXEL_STAT || cursor_pixel ||
           frames_since_pixel_section < UI_IDLE_FRAMES;
}

bool PathDebugNode::shows_reference() const {
    return backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_REFERENCE) ||
           backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_DIFFERENCE) ||
           backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_SPLIT_REFERENCE);
}

bool PathDebugNode::has_reference() const {
    return reference_input_connected || reference_loaded;
}

uint32_t PathDebugNode::selection_mode() const {
    if (!has_selection) {
        return PATH_DEBUG_SELECT_NONE;
    }
    return select_kind == SELECT_POINT
               ? PATH_DEBUG_SELECT_POINT
               : PATH_DEBUG_SELECT_THROUGH + static_cast<uint32_t>(region_match);
}

bool PathDebugNode::selects_point() const {
    return has_selection && select_kind == SELECT_POINT;
}

bool PathDebugNode::selection_visible() const {
    return has_selection && !selected_offscreen;
}

bool PathDebugNode::point_hidden() const {
    return maps_at_point() && selected_pos_valid && point_behind_surface;
}

const char* PathDebugNode::modifier_name() const {
    return MODIFIER_NAMES.at(modifier);
}

bool PathDebugNode::maps_at_point() const {
    return selects_point() && !selected_offscreen;
}

bool PathDebugNode::bsdf_comparable() const {
    return maps_at_point() && wi_mode == static_cast<int32_t>(PATH_DEBUG_WI_CAMERA);
}

uint32_t PathDebugNode::shown_panel_mask() const {
    if (!maps_at_point()) {
        return 0;
    }
    uint32_t mask = 0;
    for (uint32_t channel = 0; channel < PATH_DEBUG_MAP_CHANNEL_COUNT; channel++) {
        mask |= panel_enabled[channel] ? (1u << channel) : 0;
    }
    return mask;
}

int32_t PathDebugNode::label_px(const uint32_t size) const {
    return static_cast<int32_t>(std::lround(static_cast<float>(size) * label_scale));
}

int32_t PathDebugNode::panel_left() const {
    return label_area.x - panel_px - label_px(PATH_DEBUG_PANEL_MARGIN);
}

int32_t PathDebugNode::panel_top(const uint32_t slot) const {
    return label_px(PATH_DEBUG_PANEL_MARGIN) +
           (static_cast<int32_t>(slot) * (panel_px + label_px(PATH_DEBUG_PANEL_GAP))) -
           panel_scroll;
}

std::optional<PathDebugNode::PanelHit> PathDebugNode::panel_at(const float2& at) const {
    const auto left = static_cast<float>(panel_left());
    const auto size = static_cast<float>(panel_px);
    if (params.map_panel_mask == 0 || at.x < left || at.x >= left + size) {
        return std::nullopt;
    }
    uint32_t slot = 0;
    for (uint32_t channel = 0; channel < PATH_DEBUG_MAP_CHANNEL_COUNT; channel++) {
        if ((params.map_panel_mask & (1u << channel)) == 0) {
            continue;
        }
        const auto top = static_cast<float>(panel_top(slot++));
        if (at.y < top || at.y >= top + size) {
            continue;
        }
        const auto close = static_cast<float>(label_px(PATH_DEBUG_PANEL_CLOSE));
        const uint2 bin =
            min(uint2((at - float2(left, top)) / size * static_cast<float>(params.map_res)),
                uint2(params.map_res - 1));
        return PanelHit{channel, (bin.y * params.map_res) + bin.x,
                        at.y < top + close && at.x >= left + size - close};
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
