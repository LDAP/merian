#include "merian-graph/nodes/path_debug/path_debug.hpp"

#include "merian-shaders/debug/path_record.hpp"
#include "merian/shader/shader_compile_context.hpp"
#include "merian/vk/imgui/imgui_colormaps.hpp"
#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"
#include "merian/vk/utils/image_export.hpp"
#include "merian/vk/utils/profiler.hpp"

#include "merian/io/image_io.hpp"

#include <fmt/format.h>
#include <imgui.h>

#include <bit>
#include <cctype>
#include <cmath>

namespace merian {

namespace {
constexpr const char* SELECT_MODULE = "merian-graph/nodes/path_debug/select.slang";
constexpr const char* THRESHOLD_MODULE = "merian-graph/nodes/path_debug/threshold.slang";
constexpr const char* COLLECT_MODULE = "merian-graph/nodes/path_debug/collect.slang";
constexpr const char* DRAW_MODULE = "merian-graph/nodes/path_debug/draw.slang";
constexpr const char* COMPOSE_MODULE = "merian-graph/nodes/path_debug/compose.slang";
constexpr const char* STATS_MODULE = "merian-graph/nodes/path_debug/stats.slang";
constexpr const char* MAP_REDUCE_MODULE = "merian-graph/nodes/path_debug/map_reduce.slang";
constexpr const char* MAP_VIEW_MODULE = "merian-graph/nodes/path_debug/map_view.slang";
constexpr const char* BSDF_MAP_MODULE = "merian-graph/nodes/path_debug/bsdf_map.slang";
constexpr const char* SPHERE_MODULE = "merian-graph/nodes/path_debug/sphere_view.slang";
constexpr const char* HEAT_DECAY_MODULE = "merian-graph/nodes/path_debug/heat_decay.slang";
constexpr const char* HEAT_SPLAT_MODULE = "merian-graph/nodes/path_debug/heat_splat.slang";
constexpr const char* FINALIZE_MODULE = "merian-graph/nodes/path_debug/finalize.slang";
constexpr const char* FILTER_RENDER_MODULE = "merian-graph/nodes/path_debug/filter_render.slang";

constexpr std::array<const char*, 13> STANDALONE_MODULES = {
    SELECT_MODULE,     THRESHOLD_MODULE,  COLLECT_MODULE,      DRAW_MODULE,   COMPOSE_MODULE,
    STATS_MODULE,      MAP_REDUCE_MODULE, MAP_VIEW_MODULE,     SPHERE_MODULE, HEAT_DECAY_MODULE,
    HEAT_SPLAT_MODULE, FINALIZE_MODULE,   FILTER_RENDER_MODULE};

// mirror of path_record_pack_rgb9e5
float unpacked_luminance(const uint32_t packed) {
    const float scale = std::exp2(static_cast<float>(static_cast<int>(packed >> 27) - 15 - 9));
    return scale * ((0.2126f * static_cast<float>(packed & 511u)) +
                    (0.7152f * static_cast<float>((packed >> 9) & 511u)) +
                    (0.0722f * static_cast<float>((packed >> 18) & 511u)));
}

// event letters of the filter language: side then scattering, as in <RD>
std::string vertex_class_name(const uint32_t lobe) {
    const uint32_t smoothness =
        lobe & (PATH_DEBUG_CLS_DELTA | PATH_DEBUG_CLS_GLOSSY | PATH_DEBUG_CLS_DIFFUSE);
    const char* cls = smoothness == PATH_DEBUG_CLS_DELTA     ? "S"
                      : smoothness == PATH_DEBUG_CLS_GLOSSY  ? "G"
                      : smoothness == PATH_DEBUG_CLS_DIFFUSE ? "D"
                                                             : "U";
    return std::string((lobe & PATH_DEBUG_LOBE_TRANSMISSION) != 0 ? "T" : "R") + cls;
}

// The path's own expression, ready to paste into the filter.
std::string path_expression(const uint32_t classes, const uint32_t events) {
    std::string expression = "C";
    const uint32_t shown = std::min(events, 8u);
    for (uint32_t e = 0; e < shown; e++) {
        expression += "<" + vertex_class_name((classes >> (4 * e)) & 0xFu) + ">";
    }
    return shown < events ? expression + ".*" : expression + "<L.>";
}

constexpr uint32_t PANEL_MARGIN = 10;
constexpr uint32_t PANEL_GAP = 26;

constexpr std::array<const char*, 7> VIEW_NAMES = {
    "sampled density", "contribution",  "contribution (color)", "mean per sample",
    "BSDF pdf",        "density / pdf", "sampling z-score"};

void shadowed_text(ImDrawList* const dl, const ImVec2 pos, const char* text) {
    dl->AddText(ImVec2(pos.x + 1.f, pos.y + 1.f), IM_COL32(0, 0, 0, 200), text);
    dl->AddText(pos, IM_COL32(235, 235, 235, 255), text);
}
} // namespace

DeviceSupportInfo PathDebugNode::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo support{true};
    for (const char* module : STANDALONE_MODULES) {
        const auto composition = SlangComposition::create();
        composition->add_module_from_path(module, true);
        support = support & SlangProgram::create(query_info.compile_context, composition)
                                .get()
                                ->query_device_support(query_info);
    }
    const auto composition = Scene::query_device_support_composition(query_info);
    composition->add_module_from_path(BSDF_MAP_MODULE, true);
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
    draw_kernel.emplace(context, allocator, compile_context, DRAW_MODULE, spec_info);
    map_reduce_kernel.emplace(context, allocator, compile_context, MAP_REDUCE_MODULE, spec_info);
    map_view_kernel.emplace(context, allocator, compile_context, MAP_VIEW_MODULE, spec_info);
    heat_decay_kernel.emplace(context, allocator, compile_context, HEAT_DECAY_MODULE, spec_info);
    heat_splat_kernel.emplace(context, allocator, compile_context, HEAT_SPLAT_MODULE, spec_info);
    finalize_kernel.emplace(context, allocator, compile_context, FINALIZE_MODULE, spec_info);
    filter_render_kernel.emplace(context, allocator, compile_context, FILTER_RENDER_MODULE,
                                 spec_info);
    update_gbuffer_kernels();

    state_buffer = allocator->create_buffer(vk::DeviceSize{STATE_UINTS} * 4,
                                            vk::BufferUsageFlagBits::eStorageBuffer |
                                                vk::BufferUsageFlagBits::eTransferDst |
                                                vk::BufferUsageFlagBits::eTransferSrc,
                                            MemoryMappingType::NONE, "path_debug state");
    stats_buffer = allocator->create_buffer(vk::DeviceSize{STATS_UINTS} * 4,
                                            vk::BufferUsageFlagBits::eStorageBuffer |
                                                vk::BufferUsageFlagBits::eTransferDst |
                                                vk::BufferUsageFlagBits::eTransferSrc,
                                            MemoryMappingType::NONE, "path_debug stats");
    // sized for the maximum resolution so changing it never reallocates mid-flight
    maps_buffer = allocator->create_buffer(vk::DeviceSize{MAX_MAP_RES} * MAX_MAP_RES * 7 * 4,
                                           vk::BufferUsageFlagBits::eStorageBuffer |
                                               vk::BufferUsageFlagBits::eTransferDst |
                                               vk::BufferUsageFlagBits::eTransferSrc,
                                           MemoryMappingType::NONE, "path_debug maps");

    ::ImGuiContext* const prev_imgui_ctx = ImGui::GetCurrentContext();
    imgui_ctx = std::make_shared<ImGuiContext>();
    imgui_renderer = std::make_shared<ImGuiRenderer>(context, allocator, imgui_ctx);
    imgui_backend = std::make_shared<ImGuiMerianBackend>(imgui_ctx);
    ImGui::SetCurrentContext(prev_imgui_ctx);
}

// The kernels that read the gbuffer behind the merian_path_debug_gbuffer link constant must be
// built with the constant already right: version bumps do not reliably reach a built pipeline.
void PathDebugNode::update_gbuffer_kernels() {
    const bool connected = gbuffer_connected;
    const auto make = [connected](const char* module) {
        const auto composition = SlangComposition::create();
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
    sphere_kernel.emplace(
        context, allocator, compile_context, [make]() { return make(SPHERE_MODULE); }, spec_info);
}

void PathDebugNode::ensure_bsdf_pipeline(const SceneHandle& scene) {
    if (bsdf_composition) {
        return;
    }
    bsdf_composition = SlangComposition::create();
    bsdf_composition->add_composition(scene->get_composition());
    bsdf_composition->add_module_from_path(BSDF_MAP_MODULE, true);

    bsdf_program = SlangProgram::create(compile_context, bsdf_composition);
    bsdf_entry_point = SlangProgramEntryPoint::create(bsdf_program, "main");
    bsdf_pipeline = Versioned<Pipeline>([this] {
        const auto ep = bsdf_entry_point.get();
        return ComputePipeline::create(ep->get_pipeline_layout(context), ep->specialize());
    });
    bsdf_pipeline.depends_on(bsdf_entry_point);
    bsdf_globals = Versioned<ShaderObject>(
        [this] { return bsdf_entry_point.get()->create_global_shader_object(context, allocator); });
    bsdf_globals.depends_on(bsdf_entry_point);
}

std::vector<InputConnectorDescriptor> PathDebugNode::describe_inputs() {
    return {
        {"scene", con_scene},
        {"records", con_records, ConnectorAccess::compute_read_write},
        {"src", con_src, ConnectorAccess::compute_read},
        {"gbuffer", con_gbuffer, ConnectorAccess::compute_read, 0, true},
        {"controller", con_controller, {}, 0, true},
        {"window", con_window, {}, 0, true},
    };
}

std::vector<OutputConnectorDescriptor>
PathDebugNode::describe_outputs(const NodeIOLayout& io_layout) {
    grid_slots_log2 = std::clamp(grid_slots_log2, 8, 26);
    const vk::ImageCreateInfo src_info = io_layout[con_src]->get_create_info_or_throw();
    extent = src_info.extent;

    // storage for the compute passes plus color attachment for the label overlay
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

    overlay_buffer = allocator->create_buffer(vk::DeviceSize{extent.width} * extent.height * 4,
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              MemoryMappingType::NONE, "path_debug overlay");
    moments_buffer = allocator->create_buffer(vk::DeviceSize{extent.width} * extent.height * 6 * 4,
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                                  vk::BufferUsageFlagBits::eTransferDst,
                                              MemoryMappingType::NONE, "path_debug moments");
    filtered_buffer = allocator->create_buffer(
        vk::DeviceSize{extent.width} * extent.height * 4 * 4 * 2,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::NONE, "path_debug filtered");
    grid_buffer = allocator->create_buffer(
        (vk::DeviceSize{2} << static_cast<uint32_t>(grid_slots_log2)) * 4,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::NONE, "path_debug heat grid");

    return {{"out", con_out, ConnectorAccess::compute_write | ConnectorAccess::color_attachment},
            {"filtered", con_filtered, ConnectorAccess::compute_write},
            {"filtered_b", con_filtered_b, ConnectorAccess::compute_write}};
}

PathDebugNode::NodeStatusFlags
PathDebugNode::on_connected(const NodeIOLayout& io_layout,
                            [[maybe_unused]] const NodeIO& io,
                            const NodeConnectionInfo& info,
                            [[maybe_unused]] Submission& submission) {
    if (io_layout.is_connected(con_gbuffer) != gbuffer_connected) {
        gbuffer_connected = !gbuffer_connected;
        update_gbuffer_kernels();
    }
    bsdf_composition = nullptr;
    maps_dirty = true;
    heat_dirty = true;

    const PathRecordCapacities capacities = path_record_capacities(io[con_records]->get_size());
    matches_buffer = allocator->create_buffer(
        std::max<vk::DeviceSize>(vk::DeviceSize{(capacities.path_capacity + 31) / 32} * 4 * 2, 8),
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        MemoryMappingType::NONE, "path_debug matches");

    io_layout.register_event_listener(
        "/graph/reload_shaders", [this](const GraphEvent::Info&, const GraphEvent::Data& force) {
            for (auto* kernel : {&select_kernel, &threshold_kernel, &collect_kernel, &draw_kernel,
                                 &stats_kernel, &map_reduce_kernel, &map_view_kernel,
                                 &sphere_kernel, &compose_kernel, &heat_decay_kernel,
                                 &heat_splat_kernel, &filter_render_kernel, &finalize_kernel}) {
                (*kernel)->reload(std::any_cast<bool>(force), compile_context);
            }
            if (bsdf_composition) {
                if (std::any_cast<bool>(force)) {
                    bsdf_composition->force_reload();
                } else {
                    bsdf_composition->reload(compile_context->get_search_path_file_loader());
                }
            }
            return true;
        });
    io_layout.register_event_listener(
        "//camera_changed,//geometry_changed,//transform_changed",
        [this](const GraphEvent::Info&, const GraphEvent::Data&) {
            moments_dirty = true; // the estimates no longer describe the shown image
            maps_dirty = true;
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

void PathDebugNode::bind_globals(const NodeIO& io) {
    const SceneHandle& scene = io[con_scene];
    const bool scene_ready = scene && scene->is_ready();
    for (auto* kernel :
         {&select_kernel, &threshold_kernel, &collect_kernel, &draw_kernel, &stats_kernel,
          &map_reduce_kernel, &map_view_kernel, &sphere_kernel, &compose_kernel, &heat_decay_kernel,
          &heat_splat_kernel, &finalize_kernel, &filter_render_kernel}) {
        auto cursor = (*kernel)->globals_cursor();
        if (auto c = cursor.find("state"); c.is_valid()) {
            c = state_buffer;
        }
        if (auto c = cursor.find("overlay"); c.is_valid()) {
            c = overlay_buffer;
        }
        if (auto c = cursor.find("stats"); c.is_valid()) {
            c = stats_buffer;
        }
        if (auto c = cursor.find("maps"); c.is_valid()) {
            c = maps_buffer;
        }
        if (auto c = cursor.find("grid"); c.is_valid()) {
            c = grid_buffer;
        }
        if (auto c = cursor.find("moments"); c.is_valid()) {
            c = moments_buffer;
        }
        if (auto c = cursor.find("filtered"); c.is_valid()) {
            c = filtered_buffer;
        }
        if (auto c = cursor.find("matches"); c.is_valid()) {
            c = matches_buffer;
        }
        if (auto c = cursor.find("in_reference"); c.is_valid() && reference_texture) {
            c = reference_texture;
        }
        if (auto c = cursor.find("params"); c.is_valid()) {
            c = params;
        }
        if (auto c = cursor.find("camera"); c.is_valid() && scene_ready) {
            scene->get_active_camera()->write_to(c);
        }
        if (auto c = cursor.find("gbuffer"); c.is_valid() && gbuffer_connected) {
            c = io[con_gbuffer].r();
        }
    }
}

void PathDebugNode::handle_pick(const NodeIO& io) {
    if (io.is_connected(con_controller)) {
        const InputControllerHandle& controller = io[con_controller];
        if (controller && controller != registered_controller.lock()) {
            controller->add_listener(picker, ViewportPicker::DEFAULT_PRIORITY);
            registered_controller = controller;
        }
    }
    const WindowHandle window = io.is_connected(con_window) ? io[con_window] : nullptr;
    const vk::Extent2D image{extent.width, extent.height};

    if (const auto click = picker->take_click(image, window)) {
        selected_pixel = *click;
        selection_mode = static_cast<int32_t>(PATH_DEBUG_MODE_SELECTED_PIXEL);
        map_scope = 0;
        maps_dirty = true;
    }

    // hover state for the probe readout: panel bin when over a panel, image pixel otherwise
    params.cursor_x = PATH_DEBUG_NO_CURSOR;
    params.cursor_y = PATH_DEBUG_NO_CURSOR;
    params.probe_view = 0;
    if (const auto cursor = picker->cursor(image, window)) {
        params.cursor_x = static_cast<uint32_t>(cursor->x);
        params.cursor_y = static_cast<uint32_t>(cursor->y);
        if (panels_enabled) {
            const int32_t x0 =
                static_cast<int32_t>(extent.width - params.map_panel_size - PANEL_MARGIN);
            uint32_t panel = 0;
            for (uint32_t view = 0; view < VIEW_COUNT; view++) {
                if (!view_enabled[view]) {
                    continue;
                }
                const auto y0 = static_cast<int32_t>(PANEL_MARGIN +
                                                     panel * (params.map_panel_size + PANEL_GAP));
                if (cursor->x >= x0 &&
                    cursor->x < x0 + static_cast<int32_t>(params.map_panel_size) &&
                    cursor->y >= y0 &&
                    cursor->y < y0 + static_cast<int32_t>(params.map_panel_size)) {
                    const uint32_t bx = (cursor->x - x0) * params.map_res / params.map_panel_size;
                    const uint32_t by = (cursor->y - y0) * params.map_res / params.map_panel_size;
                    params.probe_view = view + 1;
                    params.probe_bin = std::min(by, params.map_res - 1) * params.map_res +
                                       std::min(bx, params.map_res - 1);
                    break;
                }
                panel++;
            }
        }
    }
}

PathDebugNode::NodeStatusFlags
PathDebugNode::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const CommandBufferHandle& cmd = submission.get_cmd();
    const SceneHandle& scene = io[con_scene];
    const BufferHandle records = io[con_records];

    handle_pick(io);
    load_reference(submission);

    // 1. adapt the subsampling probability to the demand observed in the last readback.
    // Starting from the capacity instead of 1: a full-rate first frame overflows the stream and
    // keeps only the paths early in dispatch order — a spatially skewed sample that an unbounded
    // accumulation downstream would never forget.
    const PathRecordCapacities capacities = path_record_capacities(records->get_size());
    if (info.get_iteration() == 0 && auto_keep_prob) {
        const float pixels = static_cast<float>(extent.width) * static_cast<float>(extent.height);
        keep_prob =
            std::clamp(0.45f * static_cast<float>(capacities.path_capacity) / std::max(pixels, 1.f),
                       1e-4f, 1.f);
    }
    bool has_records = false;
    {
        const std::scoped_lock lock(stats_mutex);
        has_records = readback_valid && latest_readback.select.paths > 0;
        if (readback_valid && latest_readback.stats[0] > (1u << 30)) {
            maps_dirty = true; // re-accumulate before per-bin counters can overflow
        }
        if (readback_valid && auto_keep_prob && !freeze &&
            selection_mode != static_cast<int32_t>(PATH_DEBUG_MODE_SELECTED_PIXEL)) {
            // whichever region fills first is the binding one: with short paths the path slots
            // run out long before the vertex blocks. A stream that truncates keeps the paths
            // early in dispatch order — a spatially skewed sample no accumulation recovers from —
            // so both fills stay below capacity.
            const float vertex_fill = static_cast<float>(latest_readback.select.vertices) /
                                      static_cast<float>(capacities.vertex_capacity);
            const float path_fill = static_cast<float>(latest_readback.select.paths) /
                                    static_cast<float>(capacities.path_capacity);
            const float fill = std::max(vertex_fill, path_fill);
            // absolute target from the keep probability that produced this readback: the demand
            // scales linearly with it, and stale readbacks then re-derive the same target instead
            // of compounding a relative step every frame
            const float keep_used = std::bit_cast<float>(latest_readback.select.keep_prob);
            if (keep_used > 0.f && (fill > 1.f || fill < 0.5f)) {
                keep_prob = std::clamp(keep_used * 0.9f / std::max(fill, 1e-3f), 1e-4f, 1.f);
            }
        }
    }

    const bool ready = scene && scene->is_ready();

    // 2. clear per-frame state; arm the record header once after (re)connect.
    // The clears overwrite what the previous iteration's dispatches (and, for the record stream,
    // the renderer of this iteration) wrote, so they need a dependency of their own.
    const auto compute_to_transfer = [&](const BufferHandle& buffer) {
        return buffer->buffer_barrier2(vk::PipelineStageFlagBits2::eComputeShader |
                                           vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
                                       vk::PipelineStageFlagBits2::eTransfer,
                                       vk::AccessFlagBits2::eShaderRead |
                                           vk::AccessFlagBits2::eShaderWrite,
                                       vk::AccessFlagBits2::eTransferWrite);
    };
    cmd->barrier({compute_to_transfer(records), compute_to_transfer(state_buffer),
                  compute_to_transfer(overlay_buffer), compute_to_transfer(matches_buffer),
                  compute_to_transfer(filtered_buffer), compute_to_transfer(stats_buffer),
                  compute_to_transfer(maps_buffer), compute_to_transfer(grid_buffer),
                  compute_to_transfer(moments_buffer)});
    if (info.get_iteration() == 0) {
        cmd->fill(records);
    }
    cmd->fill(state_buffer);
    cmd->fill(overlay_buffer);
    cmd->fill(matches_buffer);
    cmd->fill(filtered_buffer);
    if (!map_accumulate) {
        maps_dirty = true;
    }
    if (maps_dirty) {
        cmd->fill(stats_buffer);
        cmd->fill(maps_buffer);
        maps_dirty = false;
        map_frames = 0;
    }
    if (heat_dirty || info.get_iteration() == 0) {
        cmd->fill(grid_buffer);
        heat_dirty = false;
        heat_frames = 0;
    }
    if (moments_dirty || info.get_iteration() == 0) {
        cmd->fill(moments_buffer);
        moments_dirty = false;
        moments_frames = 0;
        const std::scoped_lock lock(stats_mutex);
        mean_history.clear();
        rel_error_history.clear();
    }
    const auto transfer_to_compute = [&](const BufferHandle& buffer) {
        return buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eTransferWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    };
    const auto compute_to_compute = [&](const BufferHandle& buffer) {
        return buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    };
    if (info.get_iteration() == 0) {
        cmd->barrier(transfer_to_compute(records));
    }
    cmd->barrier({transfer_to_compute(state_buffer), transfer_to_compute(overlay_buffer),
                  transfer_to_compute(stats_buffer), transfer_to_compute(maps_buffer),
                  transfer_to_compute(grid_buffer), transfer_to_compute(moments_buffer),
                  transfer_to_compute(filtered_buffer), transfer_to_compute(matches_buffer)});

    // 3. per-frame parameters
    selected_pixel.x = std::clamp(selected_pixel.x, 0, static_cast<int32_t>(extent.width) - 1);
    selected_pixel.y = std::clamp(selected_pixel.y, 0, static_cast<int32_t>(extent.height) - 1);
    params.dim_x = extent.width;
    params.dim_y = extent.height;
    params.selected_x = static_cast<uint32_t>(selected_pixel.x);
    params.selected_y = static_cast<uint32_t>(selected_pixel.y);
    params.mode = static_cast<uint32_t>(selection_mode);
    params.k = static_cast<uint32_t>(top_k);
    params.color_mode = static_cast<uint32_t>(color_mode);
    // freezing before anything was captured would keep an empty stream forever
    const bool frozen = freeze && has_records;
    params.freeze = frozen ? 1 : 0;
    params.keep_prob =
        selection_mode == static_cast<int32_t>(PATH_DEBUG_MODE_SELECTED_PIXEL) ? 1.f : keep_prob;
    params.pixel_filter = selection_mode == static_cast<int32_t>(PATH_DEBUG_MODE_SELECTED_PIXEL)
                              ? ((params.selected_y << 16) | params.selected_x)
                              : PATH_RECORD_ALL_PIXELS;
    params.max_draw = MAX_DRAW;
    map_res_log2 = std::clamp(map_res_log2, 3, 10);
    grid_slots_log2 = std::clamp(grid_slots_log2, 8, 26);
    params.map_res = 1u << static_cast<uint32_t>(map_res_log2);
    params.map_transform = static_cast<uint32_t>(map_transform);
    params.map_frame = gbuffer_connected ? static_cast<uint32_t>(map_frame) : 0;
    params.map_scope = static_cast<uint32_t>(map_scope);
    params.map_reference = static_cast<uint32_t>(map_reference);
    params.map_bounce = map_bounce < 0 ? PATH_DEBUG_ALL_BOUNCES : static_cast<uint32_t>(map_bounce);
    params.map_panel_size =
        std::clamp(params.map_panel_size, 64u, extent.height - (2 * PANEL_MARGIN));
    params.sphere_view =
        sphere_view == 0 || !gbuffer_connected ? 0 : static_cast<uint32_t>(sphere_view);
    params.heat_mode = gbuffer_connected ? static_cast<uint32_t>(heat_mode) : 0;
    params.heat_scope = static_cast<uint32_t>(heat_scope);
    heat_bounce_range.x = std::clamp(heat_bounce_range.x, 0, 30);
    heat_bounce_range.y = std::clamp(heat_bounce_range.y, heat_bounce_range.x, 30);
    params.heat_bounce_min = static_cast<uint32_t>(heat_bounce_range.x);
    params.heat_bounce_max = static_cast<uint32_t>(heat_bounce_range.y);
    params.heat_decay = 1.f - heat_alpha;
    // from the allocation, not the property: the property change is only honored at reconnect
    params.grid_slots = static_cast<uint32_t>(grid_buffer->get_size() / 4 / 2);
    params.accumulate_moments = (!frozen || moments_frames == 0) ? 1 : 0;
    params.moments_scope = static_cast<uint32_t>(moments_scope);
    params.error_view = static_cast<uint32_t>(error_view);
    if (params.accumulate_moments != 0 && ready) {
        moments_frames++;
    }
    params.filter_len_min = filter_enabled ? static_cast<uint32_t>(filter_length.x) : 0;
    params.filter_len_max = filter_enabled ? static_cast<uint32_t>(filter_length.y) : 0xFFFFFFFFu;
    params.filter_count = (filter_enabled && filter_valid) ? filter_token_count : 0;
    params.filter2_tokens_0 = filter_words_b[0];
    params.filter2_tokens_1 = filter_words_b[1];
    params.filter2_tokens_2 = filter_words_b[2];
    params.filter2_tokens_3 = filter_words_b[3];
    params.filter2_count = (filter_enabled && filter_valid_b) ? filter_token_count_b : 0;
    params.filter_method_mask = filter_enabled ? filter_method_mask : 0;
    params.filter_method_all = filter_method_mode == 1 ? 1 : 0;
    params.filter_material = (filter_enabled && filter_material >= 0)
                                 ? static_cast<uint32_t>(filter_material)
                                 : PATH_RECORD_MATERIAL_NONE;
    params.render_mode = static_cast<uint32_t>(render_mode);
    params.ab_split = ab_split;
    params.ab_metrics = reference_loaded ? 1 : 0;
    params.ab_scale = (render_mode == static_cast<int32_t>(PATH_DEBUG_RENDER_FILTERED) ||
                       render_mode >= static_cast<int32_t>(PATH_DEBUG_RENDER_FILTERED_B))
                          ? filtered_exposure
                          : ab_scale;
    // flipping A and B in place makes small differences far easier to spot than a side by side
    if (ab_flip && (render_mode == static_cast<int32_t>(PATH_DEBUG_RENDER_REFERENCE) ||
                    render_mode == static_cast<int32_t>(PATH_DEBUG_RENDER_SPLIT))) {
        const auto phase = static_cast<uint64_t>(info.get_elapsed() * ab_flip_hz);
        params.render_mode =
            (phase & 1u) == 0 ? PATH_DEBUG_RENDER_SRC : PATH_DEBUG_RENDER_REFERENCE;
    }
    params.focus_path = focus_path < 0 ? PATH_DEBUG_NO_FOCUS : static_cast<uint32_t>(focus_path);

    uint32_t view_mask = 0;
    for (uint32_t view = 0; view < VIEW_COUNT; view++) {
        view_mask |= view_enabled[view] ? (1u << view) : 0;
    }
    params.map_view = view_mask;

    bind_globals(io);

    const uint32_t path_groups = (capacities.path_capacity + PATH_DEBUG_WG - 1) / PATH_DEBUG_WG;
    const uint32_t map_bins = params.map_res * params.map_res;

    // 4. selection, maps, heat, compose, panels, finalize
    if (ready) {
        {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "select");
            const auto pipe = select_kernel->bind(io, info, submission);
            cmd->dispatch(path_groups, 1, 1);
        }
        // every later pass reads the match bits the select pass just published
        cmd->barrier(compute_to_compute(matches_buffer));
        {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "filter render");
            const auto pipe = filter_render_kernel->bind(io, info, submission);
            cmd->dispatch(path_groups, 1, 1);
        }
        // accumulate maps once per capture; frozen records must not be counted repeatedly
        if (!frozen || map_frames == 0) {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "stats");
            const auto pipe = stats_kernel->bind(io, info, submission);
            cmd->dispatch(path_groups, 1, 1);
            map_frames++;
        }
        if (gbuffer_connected) {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "bsdf map");
            ensure_bsdf_pipeline(scene);
            const auto ep = bsdf_entry_point.get();
            const auto pipe = bsdf_pipeline.get();
            const auto globals = bsdf_globals.get();
            auto cursor = globals->get_cursor();
            if (auto c = cursor.find("maps"); c.is_valid()) {
                c = maps_buffer;
            }
            if (auto c = cursor.find("stats"); c.is_valid()) {
                c = stats_buffer;
            }
            if (auto c = cursor.find("gbuffer"); c.is_valid()) {
                c = io[con_gbuffer].r();
            }
            if (auto c = cursor.find("params"); c.is_valid()) {
                c = params;
            }
            cmd->bind(pipe);
            ep->bind("scene", scene->get_shader_object(), cmd, pipe,
                     info.get_shader_object_allocator());
            ep->bind_global(globals, cmd, pipe, info.get_shader_object_allocator());
            cmd->dispatch(vk::Extent3D{params.map_res, params.map_res, 1}, 16, 16);
        }
        cmd->barrier({compute_to_compute(state_buffer), compute_to_compute(stats_buffer),
                      compute_to_compute(maps_buffer)});
        {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "reduce+threshold");
            const auto reduce_pipe = map_reduce_kernel->bind(io, info, submission);
            cmd->dispatch((map_bins + PATH_DEBUG_WG - 1) / PATH_DEBUG_WG, 1, 1);
            const auto pipe = threshold_kernel->bind(io, info, submission);
            cmd->dispatch(1, 1, 1);
        }
        cmd->barrier({compute_to_compute(state_buffer), compute_to_compute(stats_buffer)});
        {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "collect+draw");
            const auto collect_pipe = collect_kernel->bind(io, info, submission);
            cmd->dispatch(path_groups, 1, 1);
            cmd->barrier(compute_to_compute(state_buffer));
            const auto pipe = draw_kernel->bind(io, info, submission);
            cmd->dispatch(((MAX_DRAW * MAX_SEGMENTS) + PATH_DEBUG_WG - 1) / PATH_DEBUG_WG, 1, 1);
        }
        if (params.sphere_view != 0) {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "sphere");
            const auto pipe = sphere_kernel->bind(io, info, submission);
            const auto span = static_cast<uint32_t>(
                std::min((2.f * params.sphere_radius * static_cast<float>(extent.height)) + 64.f,
                         static_cast<float>(std::max(extent.width, extent.height))));
            cmd->dispatch(vk::Extent3D{span, span, 1}, 16, 16);
        }
        // heat grid: the decay pass always runs, since it also publishes the display maximum
        // (with factor 1 while frozen, so the field only re-normalizes); the splat is skipped
        // once frozen records were accumulated, they must not count twice
        if (params.heat_mode != 0) {
            MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "heat");
            const bool splat = !frozen || heat_frames == 0;
            if (frozen && !splat) {
                params.heat_decay = 1.f;
            }
            {
                const auto pipe = heat_decay_kernel->bind(io, info, submission);
                cmd->dispatch(params.grid_slots / PATH_DEBUG_WG, 1, 1);
            }
            if (splat) {
                cmd->barrier(compute_to_compute(grid_buffer));
                const auto pipe = heat_splat_kernel->bind(io, info, submission);
                cmd->dispatch(path_groups, 1, 1);
                heat_frames++;
            }
        }
        cmd->barrier({compute_to_compute(overlay_buffer), compute_to_compute(stats_buffer),
                      compute_to_compute(maps_buffer), compute_to_compute(grid_buffer),
                      compute_to_compute(state_buffer), compute_to_compute(moments_buffer),
                      compute_to_compute(filtered_buffer)});
    }
    {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "compose");
        const auto pipe = compose_kernel->bind(io, info, submission);
        cmd->dispatch(extent, 16, 16);
    }
    if (ready && panels_enabled && view_mask != 0) {
        MERIAN_PROFILE_SCOPE_GPU(info.get_profiler(), cmd, "map panels");
        cmd->barrier(io[con_out]->barrier2(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
            vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader,
            vk::PipelineStageFlagBits2::eComputeShader));
        const auto pipe = map_view_kernel->bind(io, info, submission);
        cmd->dispatch(vk::Extent3D{params.map_panel_size, params.map_panel_size,
                                   static_cast<uint32_t>(std::popcount(view_mask))},
                      16, 16);
    }

    // 5. label overlay on top of everything the compute passes wrote
    {
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

    if (export_next) {
        export_next = false;
        cmd->barrier(io[con_out]->barrier2(
            vk::ImageLayout::eTransferSrcOptimal, {}, vk::AccessFlagBits2::eTransferRead,
            vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eTransfer));
        image_export(
            allocator, submission, info.get_profiler(), io[con_out],
            vk::Extent2D{extent.width, extent.height},
            std::filesystem::absolute(
                export_path + image_format_extension(IMAGE_EXPORT_FORMATS.at(export_format))),
            IMAGE_EXPORT_FORMATS.at(export_format),
            export_metadata ? info.get_metadata() : ImageMetadata{});
    }

    if (ready) {
        // 6. stats readback; the maps stats must be copied before finalize zeroes the maxima
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
        cmd->barrier(state_buffer->buffer_barrier2(
            vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eTransferRead));
        cmd->copy(state_buffer, readback,
                  {vk::BufferCopy{0, 0, sizeof(SelectStats) + sizeof(Probe)},
                   vk::BufferCopy{sizeof(uint32_t) * PATH_DEBUG_STATE_FOCUS,
                                  offsetof(Readback, focus), sizeof(FocusPath)},
                   vk::BufferCopy{sizeof(uint32_t) * PATH_DEBUG_STATE_TOP, offsetof(Readback, top),
                                  sizeof(TopPath) * TOP_COUNT}});

        cmd->barrier(readback->buffer_barrier2(
            vk::PipelineStageFlagBits2::eTransfer, vk::PipelineStageFlagBits2::eHost,
            vk::AccessFlagBits2::eTransferWrite, vk::AccessFlagBits2::eHostRead));
        submission.sync_to_cpu([this, readback]() {
            const Readback data = *readback->get_memory()->map_as<Readback>();
            readback->get_memory()->unmap();
            const std::scoped_lock lock(stats_mutex);
            latest_readback = data;
            readback_valid = true;
            mean_history.push_back(std::bit_cast<float>(data.probe.selected_mean));
            rel_error_history.push_back(std::bit_cast<float>(data.probe.selected_rel_error));
            while (mean_history.size() > 512) {
                mean_history.pop_front();
                rel_error_history.pop_front();
            }
        });
    }

    return {};
}

// --- UI ---

void PathDebugNode::draw_overlay() {
    ImDrawList* const dl = ImGui::GetBackgroundDrawList();
    const float w = static_cast<float>(extent.width);

    const std::scoped_lock lock(stats_mutex);
    const auto& stats = latest_readback.stats;
    const auto stat_f = [&](const uint32_t i) { return std::bit_cast<float>(stats[i]); };

    if (panels_enabled) {
        const float panel = static_cast<float>(params.map_panel_size);
        const float x0 = w - panel - PANEL_MARGIN;
        uint32_t panel_index = 0;
        for (uint32_t view = 0; view < VIEW_COUNT; view++) {
            if (!view_enabled[view]) {
                continue;
            }
            const float y0 = PANEL_MARGIN + (static_cast<float>(panel_index) * (panel + PANEL_GAP));
            const float label_y = y0 + panel + 3.f;
            dl->AddRect(ImVec2(x0 - 1.f, y0 - 1.f), ImVec2(x0 + panel + 1.f, y0 + panel + 1.f),
                        IM_COL32(90, 90, 90, 255));

            const float shared_max =
                std::max(stat_f(PATH_DEBUG_STATS_MAX_DENSITY), stat_f(PATH_DEBUG_STATS_MAX_PDF));
            std::string label;
            switch (view) {
            case 0:
                label =
                    fmt::format("{}  shared scale, max {:.3g}/sr", VIEW_NAMES[view], shared_max);
                break;
            case 1:
                label = fmt::format("{}  max {:.3g}/sr", VIEW_NAMES[view],
                                    stat_f(PATH_DEBUG_STATS_MAX_CONTRIB));
                break;
            case 2:
                label = fmt::format("{}  exposure {:.2g}", VIEW_NAMES[view], params.map_exposure);
                break;
            case 3:
                label = fmt::format("{}  max {:.3g}", VIEW_NAMES[view],
                                    stat_f(PATH_DEBUG_STATS_MAX_MEAN));
                break;
            case 4:
                label =
                    fmt::format("{}  shared scale, max {:.3g}  integral {:.3f}", VIEW_NAMES[view],
                                shared_max, stat_f(PATH_DEBUG_STATS_PDF_INTEGRAL));
                break;
            case 5:
                label = fmt::format("{}  blue 1/4x  white 1x  red 4x", VIEW_NAMES[view]);
                break;
            default:
                label = stats[PATH_DEBUG_STATS_CHI2_DOF] > 0
                            ? fmt::format("{}  +-4 sigma  chi2/dof {:.2f}", VIEW_NAMES[view],
                                          stat_f(PATH_DEBUG_STATS_CHI2) /
                                              static_cast<float>(stats[PATH_DEBUG_STATS_CHI2_DOF]))
                            : fmt::format("{}  +-4 sigma  chi2/dof n/a", VIEW_NAMES[view]);
                break;
            }
            const float text_w = ImGui::CalcTextSize(label.c_str()).x;
            shadowed_text(dl, ImVec2(std::min(x0, w - PANEL_MARGIN - text_w), label_y),
                          label.c_str());

            // colorbar in the label strip, right-aligned; heat views are log over 3 decades
            if (view != PATH_DEBUG_VIEW_CONTRIB_COLOR) {
                const float bar_w = 70.f;
                const float bar_x = x0 + panel - bar_w;
                const float bar_y = label_y + 12.f;
                constexpr int SEGMENTS = 10;
                for (int s = 0; s < SEGMENTS; s++) {
                    const float t0 = static_cast<float>(s) / SEGMENTS;
                    const float t1 = static_cast<float>(s + 1) / SEGMENTS;
                    const bool diverging =
                        view == PATH_DEBUG_VIEW_RATIO || view == PATH_DEBUG_VIEW_ZSCORE;
                    const ImU32 c0 =
                        diverging ? imgui_colormap_diverging(t0) : imgui_colormap_turbo(t0);
                    const ImU32 c1 =
                        diverging ? imgui_colormap_diverging(t1) : imgui_colormap_turbo(t1);
                    dl->AddRectFilledMultiColor(ImVec2(bar_x + (t0 * bar_w), bar_y),
                                                ImVec2(bar_x + (t1 * bar_w), bar_y + 7.f), c0, c1,
                                                c1, c0);
                }
            }
            panel_index++;
        }
        if (panel_index > 0) {
            const float y_footer =
                PANEL_MARGIN + (static_cast<float>(panel_index) * (panel + PANEL_GAP)) + 2.f;
            static constexpr std::array<const char*, 3> TRANSFORM_NAMES = {"octahedral", "lat-long",
                                                                           "hemisphere"};
            const std::string footer =
                fmt::format("{}x{} {} | {} frame | {} samples | heat: log, 3 decades",
                            params.map_res, params.map_res, TRANSFORM_NAMES[params.map_transform],
                            params.map_frame == 1 ? "local" : "world", stats[1]);
            const float footer_w = ImGui::CalcTextSize(footer.c_str()).x;
            shadowed_text(dl, ImVec2(std::min(x0, w - PANEL_MARGIN - footer_w), y_footer),
                          footer.c_str());
        }
    }

    if (params.mode == PATH_DEBUG_MODE_SELECTED_PIXEL) {
        const auto x = static_cast<float>(params.selected_x);
        const auto y = static_cast<float>(params.selected_y);
        shadowed_text(dl, ImVec2(x + 14.f, y - 18.f),
                      fmt::format("({}, {})", params.selected_x, params.selected_y).c_str());
    }

    // hover readout (probe values lag the cursor by the frames in flight)
    if (params.cursor_x != PATH_DEBUG_NO_CURSOR && latest_readback.probe.flags != 0) {
        const auto& probe = latest_readback.probe;
        std::string text;
        if (probe.flags == 2 && params.probe_view != 0) {
            text = fmt::format("{}\nbin {},{}  n {}\ndensity {:.3g}/sr  pdf {:.3g}\nmean L {:.3g}",
                               VIEW_NAMES[params.probe_view - 1], params.probe_bin % params.map_res,
                               params.probe_bin / params.map_res, probe.count,
                               std::bit_cast<float>(probe.a), std::bit_cast<float>(probe.b),
                               std::bit_cast<float>(probe.c));
        } else if (probe.flags == 1) {
            text = fmt::format("({}, {})  n {}\nmean L {:.4g}\nrel err {:.3f}  var {:.3g}",
                               params.cursor_x, params.cursor_y, probe.count,
                               std::bit_cast<float>(probe.a), std::bit_cast<float>(probe.b),
                               std::bit_cast<float>(probe.c));
        }
        if (!text.empty()) {
            const float tx = std::min(static_cast<float>(params.cursor_x) + 16.f, w - 260.f);
            const float ty = std::min(static_cast<float>(params.cursor_y) + 16.f,
                                      static_cast<float>(extent.height) - 70.f);
            const ImVec2 size = ImGui::CalcTextSize(text.c_str());
            dl->AddRectFilled(ImVec2(tx - 5.f, ty - 3.f),
                              ImVec2(tx + size.x + 5.f, ty + size.y + 3.f),
                              IM_COL32(15, 15, 18, 215), 4.f);
            shadowed_text(dl, ImVec2(tx, ty), text.c_str());
        }
    }
}

// Lists the paths currently selected (brightest first for the firefly modes) and dissects the
// isolated one vertex by vertex. Freeze first: the record indices only survive while the stream
// is not rewritten.
void PathDebugNode::draw_inspector() {
    if (!freeze) {
        ImGui::TextDisabled("freeze the records to isolate a path");
    }

    if (ImGui::BeginTable("paths", 4,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY,
                          ImVec2(0.f, 130.f))) {
        ImGui::TableSetupColumn("path", ImGuiTableColumnFlags_WidthFixed, 74.f);
        ImGui::TableSetupColumn("pixel", ImGuiTableColumnFlags_WidthFixed, 84.f);
        ImGui::TableSetupColumn("luminance", ImGuiTableColumnFlags_WidthFixed, 74.f);
        ImGui::TableSetupColumn("expression", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (uint32_t t = 0; t < TOP_COUNT; t++) {
            const TopPath& entry = latest_readback.top[t];
            if (entry.index == PATH_DEBUG_NO_FOCUS) {
                break;
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool selected = focus_path == static_cast<int32_t>(entry.index);
            if (ImGui::Selectable(fmt::format("#{}##path{}", entry.index, t).c_str(), selected,
                                  ImGuiSelectableFlags_SpanAllColumns)) {
                focus_path = selected ? -1 : static_cast<int32_t>(entry.index);
            }
            ImGui::TableNextColumn();
            ImGui::Text("%u, %u", entry.pixel & 0xFFFFu, entry.pixel >> 16);
            ImGui::TableNextColumn();
            ImGui::Text("%.4g", std::bit_cast<float>(entry.luminance));
            ImGui::TableNextColumn();
            const std::string expression =
                path_expression(entry.classes, entry.sample_and_scatter & 0xFFFFu);
            ImGui::TextUnformatted(expression.c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton(fmt::format("filter##expr{}", t).c_str())) {
                filter_pattern = expression;
                filter_enabled = true;
                filter_valid = update_filter();
                filter_length = int2(0, 32);
                filter_method_mask = 0;
                filter_material = -1;
                maps_dirty = true;
                moments_dirty = true;
            }
        }
        ImGui::EndTable();
    }

    if (focus_path < 0) {
        ImGui::TextDisabled("select a path to isolate it in the image");
        return;
    }
    if (ImGui::Button("show all paths again")) {
        focus_path = -1;
        return;
    }

    const FocusPath& focus = latest_readback.focus;
    if (focus.valid == 0) {
        ImGui::TextDisabled("path #%d is not in the current record stream", focus_path);
        return;
    }
    const uint32_t vertices = focus.sample_and_count >> 16;
    ImGui::Text("pixel (%u, %u) | sample %u | frame %u | luminance %.4g", focus.pixel & 0xFFFFu,
                focus.pixel >> 16, focus.sample_and_count & 0xFFFFu,
                latest_readback.select.records_frame, std::bit_cast<float>(focus.luminance));
    ImGui::TextDisabled("seed of this path: (pixel, sample, frame) — replay it in the renderer");

    if (ImGui::BeginTable("vertices", 8,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY,
                          ImVec2(0.f, 200.f))) {
        ImGui::TableSetupColumn("i", ImGuiTableColumnFlags_WidthFixed, 24.f);
        ImGui::TableSetupColumn("class", ImGuiTableColumnFlags_WidthFixed, 44.f);
        ImGui::TableSetupColumn("pdf", ImGuiTableColumnFlags_WidthFixed, 76.f);
        ImGui::TableSetupColumn("throughput", ImGuiTableColumnFlags_WidthFixed, 82.f);
        ImGui::TableSetupColumn("contribution", ImGuiTableColumnFlags_WidthFixed, 86.f);
        ImGui::TableSetupColumn("sampled by", ImGuiTableColumnFlags_WidthFixed, 74.f);
        ImGui::TableSetupColumn("material", ImGuiTableColumnFlags_WidthFixed, 60.f);
        ImGui::TableSetupColumn("position", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (uint32_t v = 0; v < vertices; v++) {
            const uint32_t* const words = focus.vertices.data() + (v * PATH_RECORD_VERTEX_UINTS);
            const uint32_t meta = words[7];
            const bool terminal = (meta & PATH_RECORD_FLAG_TERMINAL) != 0;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%u", v);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(terminal ? "end"
                                            : vertex_class_name((meta >> PATH_RECORD_LOBE_SHIFT) &
                                                                PATH_RECORD_LOBE_MASK)
                                                  .c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.4g", std::bit_cast<float>(words[5]));
            ImGui::TableNextColumn();
            ImGui::Text("%.4g", std::bit_cast<float>(words[6]));
            ImGui::TableNextColumn();
            ImGui::Text("%.3g", unpacked_luminance(words[4]));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(
                terminal ? ""
                         : path_record_method_name((meta >> PATH_RECORD_METHOD_SHIFT) &
                                                   PATH_RECORD_METHOD_MASK));
            ImGui::TableNextColumn();
            const uint32_t material =
                (meta >> PATH_RECORD_MATERIAL_SHIFT) & PATH_RECORD_MATERIAL_MASK;
            if (terminal || material == PATH_RECORD_MATERIAL_NONE) {
                ImGui::TextUnformatted("");
            } else {
                ImGui::Text("%u", material);
            }
            ImGui::TableNextColumn();
            const float3 pos(std::bit_cast<float>(words[0]), std::bit_cast<float>(words[1]),
                             std::bit_cast<float>(words[2]));
            // a path that left the scene ends at the far plane
            if (std::abs(pos.x) > 1e6f || std::abs(pos.y) > 1e6f || std::abs(pos.z) > 1e6f) {
                ImGui::TextDisabled("environment");
            } else {
                ImGui::Text("%.2f %.2f %.2f", pos.x, pos.y, pos.z);
            }
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("a pdf far below its neighbours, or a throughput that jumps, is where a "
                        "firefly is born");
    ImGui::TextDisabled("contribution is emission seen at this vertex, transported by the "
                        "previous row's throughput");
}

void PathDebugNode::draw_window() {
    ImGui::SetNextWindowPos(ImVec2(440.f, 40.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(600.f, 560.f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Path Debugger")) {
        ImGui::End();
        return;
    }

    const std::scoped_lock lock(stats_mutex);
    ImGui::Text("selected pixel: (%d, %d) — ctrl+click to pick", selected_pixel.x,
                selected_pixel.y);
    if (readback_valid) {
        ImGui::Text("paths %u | vertices %u | selected %u | drawn %u | keep %.3f",
                    latest_readback.select.paths, latest_readback.select.vertices,
                    latest_readback.select.selected,
                    std::min(latest_readback.select.draw_count, MAX_DRAW), keep_prob);
        const float chi2 = std::bit_cast<float>(latest_readback.stats[PATH_DEBUG_STATS_CHI2]);
        const uint32_t dof = latest_readback.stats[PATH_DEBUG_STATS_CHI2_DOF];
        const uint32_t samples = latest_readback.stats[PATH_DEBUG_STATS_SAMPLES];
        ImGui::Text("map samples %u | pdf integral %.3f | Omega %.3f sr", samples,
                    std::bit_cast<float>(latest_readback.stats[PATH_DEBUG_STATS_PDF_INTEGRAL]),
                    std::bit_cast<float>(latest_readback.stats[PATH_DEBUG_STATS_OMEGA]));
        // both references need one shared sampling support: the selected pixel's primary hit.
        // The recorded reference additionally reads low in bins the support only partly covers
        // (hemisphere rim in a full-sphere map), so the exact setup is local frame + hemisphere.
        const bool recorded = map_reference == static_cast<int32_t>(PATH_DEBUG_REFERENCE_RECORDED);
        const bool reference_applies =
            map_scope == 0 && map_bounce == 0 && (recorded || gbuffer_connected);
        if (!reference_applies) {
            ImGui::TextDisabled("chi2 needs scope 'selected pixel' and bounce 0: every sample "
                                "must draw from one shared support");
        } else if (dof > 0) {
            const float reduced = chi2 / static_cast<float>(dof);
            const float coverage =
                static_cast<float>(latest_readback.stats[PATH_DEBUG_STATS_TESTED]) /
                static_cast<float>(std::max(samples, 1u));
            ImGui::Text("records vs %s: chi2/dof = %.2f over %u bins, %.0f%% of samples tested "
                        "%s",
                        recorded ? "recorded pdf" : "BSDF pdf", reduced, dof, coverage * 100.f,
                        reduced < 1.5f ? "(consistent)" : "(inconsistent!)");
        } else {
            ImGui::TextDisabled("chi2 n/a: no bin reached the expected-count threshold yet");
        }
        const float spike_mass =
            std::bit_cast<float>(latest_readback.stats[PATH_DEBUG_STATS_SPIKE_MASS]);
        if (!recorded && latest_readback.stats[PATH_DEBUG_STATS_SPIKE_BINS] > 0) {
            ImGui::Text("quasi-delta lobes: %.1f%% of samples in %u spike bins (excluded; "
                        "the sub-sampled pdf cannot resolve them)",
                        spike_mass * 100.f, latest_readback.stats[PATH_DEBUG_STATS_SPIKE_BINS]);
        }
        ImGui::TextDisabled(recorded
                                ? "chi2/dof ~ 1 = samples follow the pdfs the sampler recorded; "
                                  "the path filter must not select on direction"
                                : "chi2/dof ~ 1 = sampling matches the pdf; orange z-score bins "
                                  "are the excluded spikes");
    }
    if (ImGui::Button(freeze ? "unfreeze" : "freeze")) {
        freeze = !freeze;
    }
    ImGui::SameLine();
    if (ImGui::Button("reset maps")) {
        maps_dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("clear heat")) {
        heat_dirty = true;
    }

    if (!readback_valid || !ImGui::BeginTabBar("tabs")) {
        ImGui::End();
        return;
    }
    const auto& stats = latest_readback.stats;
    const auto plot = [&](const char* label, const uint32_t offset, const uint32_t bins) {
        std::array<float, 64> values{};
        float vmax = 0.f;
        for (uint32_t i = 0; i < bins; i++) {
            values[i] = static_cast<float>(stats[offset + i]);
            vmax = std::max(vmax, values[i]);
        }
        ImGui::PlotHistogram(label, values.data(), static_cast<int>(bins), 0, nullptr, 0.f, vmax,
                             ImVec2(0, 60));
    };
    if (ImGui::BeginTabItem("convergence")) {
        if (!mean_history.empty()) {
            std::vector<float> means(mean_history.begin(), mean_history.end());
            std::vector<float> errs(rel_error_history.begin(), rel_error_history.end());
            ImGui::Text("selected pixel: mean L %.4g | rel err %.3f | n %u", means.back(),
                        errs.back(), latest_readback.probe.selected_count);
            ImGui::PlotLines("mean over time", means.data(), static_cast<int>(means.size()), 0,
                             nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 50));
            ImGui::PlotLines("rel. error over time", errs.data(), static_cast<int>(errs.size()), 0,
                             nullptr, 0.f, FLT_MAX, ImVec2(0, 50));
        }
        // overlaid polar profiles: matching curves = sampling, pdf and the view agree
        {
            std::array<float, THETA_BANDS> records_curve{};
            std::array<float, THETA_BANDS> pdf_curve{};
            double records_norm = 0.;
            double pdf_norm = 0.;
            for (uint32_t band = 0; band < THETA_BANDS; band++) {
                const uint32_t base = PATH_DEBUG_STATS_THETA + (3 * band);
                const float omega = std::bit_cast<float>(stats[base + 2]);
                if (omega <= 0.f) {
                    continue;
                }
                records_curve[band] = static_cast<float>(stats[base + 0]) / omega;
                pdf_curve[band] = std::bit_cast<float>(stats[base + 1]) / omega;
                records_norm += static_cast<double>(stats[base + 0]);
                pdf_norm += static_cast<double>(std::bit_cast<float>(stats[base + 1]));
            }
            ImGui::SeparatorText("density vs pdf, by angle to the pole");
            ImDrawList* const dl = ImGui::GetWindowDrawList();
            const ImVec2 origin = ImGui::GetCursorScreenPos();
            const float plot_w = ImGui::GetContentRegionAvail().x - 10.f;
            const float plot_h = 110.f;
            dl->AddRectFilled(origin, ImVec2(origin.x + plot_w, origin.y + plot_h),
                              ImGui::GetColorU32(ImGuiCol_FrameBg), 4.f);
            const auto draw_curve = [&](const std::array<float, THETA_BANDS>& curve,
                                        const double norm, const ImU32 color,
                                        const float thickness) {
                if (norm <= 0.) {
                    return;
                }
                float vmax = 0.f;
                for (const float v : curve) {
                    vmax = std::max(vmax, static_cast<float>(v / norm));
                }
                if (vmax <= 0.f) {
                    return;
                }
                std::array<ImVec2, THETA_BANDS> points;
                for (uint32_t band = 0; band < THETA_BANDS; band++) {
                    const float x =
                        origin.x + ((static_cast<float>(band) + 0.5f) / THETA_BANDS) * plot_w;
                    const float y =
                        origin.y + plot_h - 6.f -
                        (static_cast<float>(curve[band] / norm) / vmax) * (plot_h - 12.f);
                    points[band] = ImVec2(x, y);
                }
                dl->AddPolyline(points.data(), THETA_BANDS, color, ImDrawFlags_None, thickness);
            };
            // pdf as the wide reference underneath, the sampled density on top
            draw_curve(pdf_curve, pdf_norm, IM_COL32(235, 235, 235, 255), 4.f);
            draw_curve(records_curve, records_norm, IM_COL32(203, 166, 247, 255), 1.5f);
            ImGui::Dummy(ImVec2(plot_w, plot_h + 4.f));
            ImGui::TextColored(ImVec4(0.92f, 0.92f, 0.92f, 1.f), "pdf");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.80f, 0.65f, 0.97f, 1.f), "records");
            ImGui::SameLine();
            ImGui::TextDisabled("(each normalized; 0..180 deg)");
        }

        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("distributions")) {
        plot("path luminance (log2)", PATH_DEBUG_STATS_LUM, 64);
        plot("sample pdf (log2)", PATH_DEBUG_STATS_PDF, 64);
        plot("paths by scatter events", PATH_DEBUG_STATS_BOUNCE, 32);

        // where the energy actually comes from, and how efficiently each length delivers it
        {
            std::array<float, 32> contrib{};
            std::array<float, 32> mean{};
            float total = 0.f;
            for (uint32_t i = 0; i < 32; i++) {
                contrib[i] = std::bit_cast<float>(stats[PATH_DEBUG_STATS_LEN_CONTRIB + i]);
                const uint32_t n = stats[PATH_DEBUG_STATS_BOUNCE + i];
                mean[i] = n > 0 ? contrib[i] / static_cast<float>(n) : 0.f;
                total += contrib[i];
            }
            ImGui::PlotHistogram("contribution by scatter events", contrib.data(), 32, 0, nullptr,
                                 0.f, FLT_MAX, ImVec2(0, 60));
            ImGui::PlotHistogram("mean contribution per path", mean.data(), 32, 0, nullptr, 0.f,
                                 FLT_MAX, ImVec2(0, 60));
            if (total > 0.f) {
                std::string shares;
                for (uint32_t i = 0; i < 32; i++) {
                    if (contrib[i] > 0.02f * total) {
                        shares += fmt::format("{}: {:.0f}%  ", i, 100.f * contrib[i] / total);
                    }
                }
                ImGui::TextWrapped("energy share: %s", shares.c_str());
            }
        }

        ImGui::TextWrapped("directional maps render as panels in the image; configure them under "
                           "'directional maps' in the node properties");
        ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("paths")) {
        draw_inspector();
        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
    ImGui::End();
}

// Parses the light path expression into the tokens layout.slangh matches against.
bool PathDebugNode::update_filter() {
    std::array<uint32_t, 4> packed{};
    if (!compile_filter(filter_pattern, packed, filter_token_count)) {
        return false;
    }
    params.filter_tokens_0 = packed[0];
    params.filter_tokens_1 = packed[1];
    params.filter_tokens_2 = packed[2];
    params.filter_tokens_3 = packed[3];
    return true;
}

bool PathDebugNode::compile_filter(const std::string& text,
                                   std::array<uint32_t, 4>& packed,
                                   uint32_t& token_count) {
    std::array<uint32_t, PATH_DEBUG_MAX_TOKENS> tokens{};
    uint32_t count = 0;
    std::size_t i = 0;

    const auto letter_masks = [](const std::string& letters, uint32_t& scattering, uint32_t& side) {
        for (const char c : letters) {
            switch (std::toupper(static_cast<unsigned char>(c))) {
            case 'S':
                scattering |= PATH_DEBUG_CLS_DELTA;
                break;
            case 'G':
                scattering |= PATH_DEBUG_CLS_GLOSSY;
                break;
            case 'D':
                scattering |= PATH_DEBUG_CLS_DIFFUSE;
                break;
            case 'U':
                scattering |= PATH_DEBUG_CLS_UNKNOWN;
                break;
            case 'R':
                side |= PATH_DEBUG_SIDE_REFLECT;
                break;
            case 'T':
                side |= PATH_DEBUG_SIDE_TRANSMIT;
                break;
            case '.':
                break;
            default:
                return false;
            }
        }
        return true;
    };

    while (i < text.size()) {
        const char c = text[i];
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            i++;
            continue;
        }
        // camera and light anchors are implicit
        if ((c == 'C' && count == 0) ||
            (std::toupper(static_cast<unsigned char>(c)) == 'L' &&
             text.find_first_not_of(" \t", i + 1) == std::string::npos)) {
            i++;
            continue;
        }
        if (count == PATH_DEBUG_MAX_TOKENS) {
            return false;
        }

        std::string letters;
        if (c == '<' || c == '[') {
            const char closing = c == '<' ? '>' : ']';
            const std::size_t end_pos = text.find(closing, i);
            if (end_pos == std::string::npos) {
                return false;
            }
            letters = text.substr(i + 1, end_pos - i - 1);
            i = end_pos + 1;
            // <L.> is the light anchor, implicit at the end of a recorded path
            if (letters.find('L') != std::string::npos || letters.find('l') != std::string::npos) {
                continue;
            }
        } else {
            letters = std::string(1, c);
            i++;
        }

        uint32_t scattering = 0;
        uint32_t side = 0;
        if (!letter_masks(letters, scattering, side)) {
            return false;
        }
        if (scattering == 0) {
            scattering = PATH_DEBUG_CLS_DELTA | PATH_DEBUG_CLS_GLOSSY | PATH_DEBUG_CLS_DIFFUSE |
                         PATH_DEBUG_CLS_UNKNOWN;
        }
        if (side == 0) {
            side = PATH_DEBUG_SIDE_REFLECT | PATH_DEBUG_SIDE_TRANSMIT;
        }

        uint32_t repeat_min = 1;
        uint32_t repeat_max = 1;
        if (i < text.size()) {
            if (text[i] == '*') {
                repeat_min = 0;
                repeat_max = PATH_DEBUG_REPEAT_INF;
                i++;
            } else if (text[i] == '+') {
                repeat_min = 1;
                repeat_max = PATH_DEBUG_REPEAT_INF;
                i++;
            } else if (text[i] == '?') {
                repeat_min = 0;
                repeat_max = 1;
                i++;
            } else if (text[i] == '{') {
                const std::size_t end_pos = text.find('}', i);
                if (end_pos == std::string::npos) {
                    return false;
                }
                const std::string range = text.substr(i + 1, end_pos - i - 1);
                const std::size_t comma = range.find(',');
                try {
                    if (comma == std::string::npos) {
                        repeat_min = repeat_max = std::stoul(range);
                    } else {
                        repeat_min = std::stoul(range.substr(0, comma));
                        const std::string upper = range.substr(comma + 1);
                        repeat_max = upper.empty() ? PATH_DEBUG_REPEAT_INF : std::stoul(upper);
                    }
                } catch (const std::exception&) {
                    return false;
                }
                i = end_pos + 1;
            }
        }
        if (repeat_min > 7 || (repeat_max != PATH_DEBUG_REPEAT_INF && repeat_max > 14) ||
            repeat_min > repeat_max) {
            return false;
        }

        tokens[count++] = scattering | (side << 4) | (repeat_min << 6) | (repeat_max << 9);
    }

    packed[0] = tokens[0] | (tokens[1] << 16);
    packed[1] = tokens[2] | (tokens[3] << 16);
    packed[2] = tokens[4] | (tokens[5] << 16);
    packed[3] = tokens[6] | (tokens[7] << 16);
    token_count = count;
    return true;
}

void PathDebugNode::load_reference(Submission& submission) {
    if (!reference_dirty && reference_texture) {
        return;
    }
    reference_dirty = false;
    reference_loaded = false;

    ImageInfo info{};
    BlobHandle blob;
    if (!reference_path.empty()) {
        try {
            blob = image_load_f32(reference_path, info, 4);
        } catch (const std::exception& e) {
            SPDLOG_ERROR("path debugger: {}", e.what());
        }
    }
    // a 1x1 black image keeps the binding valid while nothing is loaded
    const std::array<float, 4> black{0.f, 0.f, 0.f, 1.f};
    const float* data = blob ? blob->get_data<float>() : black.data();
    const uint32_t width = blob ? static_cast<uint32_t>(info.width) : 1;
    const uint32_t height = blob ? static_cast<uint32_t>(info.height) : 1;
    reference_texture = allocator->create_texture_from_rgba32f(
        submission.get_cmd(), data, width, height, vk::SamplerAddressMode::eClampToEdge,
        vk::Filter::eLinear, vk::Filter::eLinear, "path_debug reference");
    submission.get_cmd()->barrier(
        reference_texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
    reference_loaded = blob != nullptr;
    if (reference_loaded) {
        SPDLOG_INFO("path debugger: loaded reference {} ({}x{})", reference_path, width, height);
    }
}

PathDebugNode::NodeStatusFlags PathDebugNode::properties(Properties& config) {
    NodeStatusFlags flags{};

    config.config_options("mode", selection_mode, {"top-k (image)", "selected pixel", "fireflies"},
                          Properties::OptionsStyle::COMBO,
                          "Draw the k most contributing paths of the frame, every captured path "
                          "of one pixel, or the brightest fraction of all paths.");
    config.config_int("isolate path", focus_path,
                      "Draws and dissects a single recorded path; -1 draws all. Freeze first, the "
                      "indices only survive while the stream is not rewritten.",
                      -1, static_cast<int32_t>(MAX_DRAW));
    if (selection_mode == static_cast<int32_t>(PATH_DEBUG_MODE_TOP_K)) {
        config.config_int("k", top_k, "Number of paths to draw.", 1,
                          static_cast<int32_t>(MAX_DRAW));
    } else if (selection_mode == static_cast<int32_t>(PATH_DEBUG_MODE_SELECTED_PIXEL)) {
        config.config_vec("selected pixel", selected_pixel);
    } else {
        config.config_float("firefly fraction", params.firefly_fraction,
                            "Brightest fraction of the captured paths.", 1e-5f);
        params.firefly_fraction = std::clamp(params.firefly_fraction, 1e-7f, 1.f);
    }
    config.config_bool("freeze", freeze,
                       "Stop capturing; keeps the current records for inspection.");

    if (config.st_begin_child("capture", "capture")) {
        config.config_bool("auto subsample", auto_keep_prob,
                           "Adapt the path keep-probability to the record buffer capacity.");
        if (!auto_keep_prob) {
            config.config_percent("keep probability", keep_prob);
        }
        {
            const std::scoped_lock lock(stats_mutex);
            if (readback_valid) {
                config.output_text(fmt::format(
                    "paths: {}\nvertices: {}\nselected: {}\ndrawn: {}\nkeep probability: {:.4f}",
                    latest_readback.select.paths, latest_readback.select.vertices,
                    latest_readback.select.selected,
                    std::min(latest_readback.select.draw_count, MAX_DRAW), keep_prob));
            }
        }
        config.st_end_child();
    }

    if (config.st_begin_child("maps", "directional maps")) {
        if (!gbuffer_connected) {
            config.output_text(
                "connect the gbuffer input for local frames, the BSDF pdf and the sphere view");
        }
        maps_dirty |= config.config_int(
            "resolution (log2)", map_res_log2,
            "Map resolution 8x8 (3) .. 1024x1024 (10); accumulate longer for higher resolutions.",
            3, 10);
        maps_dirty |= config.config_options(
            "transform", map_transform, {"octahedral", "lat-long", "hemisphere (equal-area)"},
            Properties::OptionsStyle::COMBO,
            "Direction to map layout; hemisphere covers only the upper half of the frame.");
        maps_dirty |= config.config_options(
            "frame", map_frame, {"world (y up)", "local shading frame"},
            Properties::OptionsStyle::COMBO,
            "Local uses the gbuffer normal of the selected pixel; only meaningful with scope "
            "'selected pixel' and bounce 0.");
        maps_dirty |= config.config_options(
            "scope", map_scope, {"selected pixel", "whole image"}, Properties::OptionsStyle::COMBO,
            "Whole image costs a few ms per frame (histogram atomics over every vertex).");
        maps_dirty |= config.config_int(
            "bounce", map_bounce, "Vertex index for the maps and pdf histogram; -1 = all.", -1, 30);
        maps_dirty |= config.config_options(
            "reference", map_reference, {"BSDF pdf", "recorded pdf"},
            Properties::OptionsStyle::COMBO,
            "z-score/ratio/chi2 reference. BSDF pdf compares the sampled density against the "
            "material at the selected pixel's primary hit. Recorded pdf tests every sampler "
            "against the density it claims per sample (E[sum 1/pdf] = draws * omega per bin); "
            "both need scope 'selected pixel' and bounce 0 so all draws share one support.");
        config.config_bool("accumulate", map_accumulate,
                           "Accumulate over frames; otherwise per-frame.");
        if (config.config_bool("reset", "Clear the accumulated maps.")) {
            maps_dirty = true;
        }
        config.st_separate("views");
        for (uint32_t view = 0; view < VIEW_COUNT; view++) {
            config.config_bool(VIEW_NAMES[view], view_enabled[view]);
        }
        config.config_bool("show panels", panels_enabled);
        int32_t panel_size = static_cast<int32_t>(params.map_panel_size);
        config.config_int("panel size", panel_size, "", 64, 1024);
        params.map_panel_size = static_cast<uint32_t>(panel_size);
        config.config_float("exposure", params.map_exposure,
                            "Shifts the log display range; scales the color view.", 0.01f);
        config.config_options("sphere view", sphere_view,
                              {"off", "sampled density", "contribution", "contribution (color)",
                               "mean per sample", "BSDF pdf", "density / pdf", "sampling z-score"},
                              Properties::OptionsStyle::COMBO,
                              "Draws the selected map on a sphere at the first surface point.");
        config.config_percent("sphere size", params.sphere_radius);
        config.st_end_child();
    }

    if (config.st_begin_child("heat", "scene heatmap")) {
        if (!gbuffer_connected) {
            config.output_text("connect the gbuffer input to enable the heatmap");
        }
        heat_dirty |= config.config_options("weight", heat_mode,
                                            {"off", "density", "contribution", "throughput"},
                                            Properties::OptionsStyle::COMBO,
                                            "Splat recorded vertices into a world-space grid and "
                                            "shade primary surfaces by the local value.");
        heat_dirty |=
            config.config_options("splat scope", heat_scope, {"all pixels", "selected pixel"},
                                  Properties::OptionsStyle::COMBO);
        heat_dirty |= config.config_vec("bounce range", heat_bounce_range,
                                        "Vertex indices splatted; 1..30 shows where first-bounce "
                                        "rays land.");
        heat_dirty |= config.config_float("cell size", params.heat_cell_size,
                                          "Grid cell edge length in world units.", 0.001f);
        params.heat_cell_size = std::max(params.heat_cell_size, 1e-4f);
        if (config.config_int("grid size (log2)", grid_slots_log2,
                              "Hash grid slot count; larger grids alias less on large scenes.", 8,
                              26)) {
            heat_dirty = true;
            flags |= NEEDS_RECONNECT; // the grid buffer is recreated on connect
        }
        config.config_percent("alpha", heat_alpha,
                              "Temporal blend: 0 accumulates forever, higher forgets faster.");
        config.config_float("exposure", params.heat_exposure, "", 0.01f);
        config.config_percent("opacity", params.heat_opacity);
        config.config_bool("smooth", params.heat_smooth);
        if (config.config_bool("clear", "Reset the accumulated heat grid.")) {
            heat_dirty = true;
        }
        config.st_end_child();
    }

    if (config.st_begin_child("display", "display")) {
        config.config_options("error overlay", error_view,
                              {"off", "relative error", "variance", "mean"},
                              Properties::OptionsStyle::COMBO,
                              "Per-pixel luminance statistics accumulated from the records; "
                              "resets on camera or scene changes.");
        if (error_view != 0) {
            config.config_float("error scale", params.error_scale, "", 0.1f);
        }
        config.config_options("statistics scope", moments_scope, {"all paths", "filtered paths"},
                              Properties::OptionsStyle::COMBO,
                              "Which paths feed the error overlay, convergence plot and hover "
                              "probe: everything recorded, or only the paths matching the "
                              "filter.");
        if (config.config_bool("reset statistics", "Clear the per-pixel moment accumulation.")) {
            moments_dirty = true;
        }
        config.config_percent("overlay alpha", params.overlay_alpha);
        config.config_options("path color", color_mode, {"luminance", "bounces", "random"},
                              Properties::OptionsStyle::COMBO);
        if (color_mode == 0) {
            config.config_float("color scale", params.color_scale,
                                "Scales luminance before the colormap.", 0.01f);
        }
        config.st_end_child();
    }

    if (config.st_begin_child("filter", "path filter")) {
        if (config.config_bool("enable", filter_enabled,
                               "Restricts every view to the paths matching the pattern.")) {
            maps_dirty = true;
            moments_dirty = true;
        }
        if (config.config_text(
                "path expression", filter_pattern, false,
                "One atom per scatter event: S specular, G glossy, D diffuse, U unclassified, "
                "R reflected, T transmitted, '.' any, <RD> combined, [SG] alternatives; with "
                "repeats * + ? {n} {n,m}. Matches the whole path, so '.*<TD>.*' means 'contains "
                "a diffuse transmission' and '.{2}' means 'exactly two events'.")) {
            filter_valid = update_filter();
            maps_dirty = true;
            moments_dirty = true;
        }
        if (!filter_valid) {
            config.output_text("pattern does not parse");
        }
        if (config.config_text(
                "path expression B", filter_pattern_b, false,
                "Second expression over the same constraints; its paths land on the "
                "'filtered_b' output for side-by-side class comparison. Empty matches "
                "everything.")) {
            filter_valid_b = compile_filter(filter_pattern_b, filter_words_b, filter_token_count_b);
        }
        if (!filter_valid_b) {
            config.output_text("pattern B does not parse");
        }
        if (config.config_vec("scatter events", filter_length,
                              "Range of sampled directions on the path (min, max).")) {
            filter_length.x = std::clamp(filter_length.x, 0, 32);
            filter_length.y = std::clamp(filter_length.y, filter_length.x, 32);
            maps_dirty = true;
            moments_dirty = true;
        }
        if (config.st_begin_child("method", "sampling method")) {
            for (uint32_t m = 0; m < PATH_RECORD_METHOD_NAMES.size(); m++) {
                bool set = (filter_method_mask & (1u << m)) != 0;
                if (config.config_bool(PATH_RECORD_METHOD_NAMES[m], set)) {
                    filter_method_mask =
                        set ? (filter_method_mask | (1u << m)) : (filter_method_mask & ~(1u << m));
                    maps_dirty = true;
                    moments_dirty = true;
                }
            }
            if (config.config_options("mode", filter_method_mode, {"any vertex", "every vertex"},
                                      Properties::OptionsStyle::COMBO,
                                      "Whether one matching scatter event suffices or all must "
                                      "match. No checked method disables the constraint.")) {
                maps_dirty = true;
                moments_dirty = true;
            }
            config.st_end_child();
        }
        if (config.config_int("material", filter_material,
                              "Keep paths hitting this material id at any scatter vertex; -1 = "
                              "off. Ids above 4094 are not recorded.",
                              -1, static_cast<int32_t>(PATH_RECORD_MATERIAL_NONE) - 1)) {
            maps_dirty = true;
            moments_dirty = true;
        }
        config.output_text(fmt::format("matched {} of {} recorded paths",
                                       readback_valid ? latest_readback.select.selected : 0,
                                       readback_valid ? latest_readback.select.paths : 0));
        config.st_end_child();
    }

    if (config.st_begin_child("ab", "A/B")) {
        config.config_options("show", render_mode,
                              {"input image", "filtered paths", "reference", "difference", "split",
                               "filtered paths B", "split filters A|B"},
                              Properties::OptionsStyle::COMBO,
                              "What the overlays are drawn on: A is the input image, B the "
                              "reference loaded from disk; the last two show the filter slots.");
        if (render_mode == static_cast<int32_t>(PATH_DEBUG_RENDER_FILTERED) ||
            render_mode >= static_cast<int32_t>(PATH_DEBUG_RENDER_FILTERED_B)) {
            config.config_float("exposure", filtered_exposure,
                                "The filtered view shows one frame of raw radiance; 0 disables "
                                "the tonemap. Accumulate the 'filtered' output for a clean image. "
                                "With 'demodulate albedo' on, the records (and this view) carry "
                                "the albedo the renderer divides out.",
                                0.01f);
        }
        if (config.config_text("reference image", reference_path, true,
                               "Any format the image loader reads; resampled to the view.")) {
            reference_dirty = true;
        }
        if (!reference_path.empty()) {
            config.output_text(reference_loaded ? "reference loaded" : "reference not loaded");
        }
        if (reference_loaded && readback_valid) {
            const uint32_t n = latest_readback.stats[PATH_DEBUG_STATS_AB_N];
            if (n > 0) {
                const float relmse =
                    std::bit_cast<float>(latest_readback.stats[PATH_DEBUG_STATS_AB_RELSQ]) /
                    static_cast<float>(n);
                const float signed_diff =
                    std::bit_cast<float>(latest_readback.stats[PATH_DEBUG_STATS_AB_SIGNED]) /
                    static_cast<float>(n);
                config.output_text(
                    fmt::format("input vs reference: relMSE {:.3e} | mean luminance diff {:+.3e}",
                                relmse, signed_diff));
            }
        }
        config.config_bool("flip A/B", ab_flip,
                           "Alternates between input and reference in place; differences are much "
                           "easier to see than side by side.");
        if (ab_flip) {
            config.config_float("flip rate (Hz)", ab_flip_hz, "", 0.1f);
        }
        if (render_mode == static_cast<int32_t>(PATH_DEBUG_RENDER_SPLIT) ||
            render_mode == static_cast<int32_t>(PATH_DEBUG_RENDER_SPLIT_FILTERS)) {
            config.config_percent("split", ab_split);
        }
        if (render_mode == static_cast<int32_t>(PATH_DEBUG_RENDER_DIFF)) {
            config.config_float("difference gain", ab_scale, "", 0.1f);
        }
        config.st_end_child();
    }

    if (config.st_begin_child("export", "export")) {
        std::ignore = config.config_text("filename", export_path, false,
                                         "Path without extension; directories are created.");
        config.config_options("format", export_format, {"PNG", "JPG", "HDR", "PFM"},
                              Properties::OptionsStyle::COMBO);
        config.config_bool("embed metadata", export_metadata,
                           "embeds the merian version and the graph config into the file (PNG, "
                           "JPG, HDR)");
        export_next |= config.config_bool("export now", "Writes the current view to disk.");
        config.st_end_child();
    }

    if (config.config_text("imgui event pattern", imgui_event_pattern, true,
                           "Graph event that carries the ImGui frame.")) {
        flags |= NEEDS_RECONNECT; // listener must be re-registered
    }

    return flags;
}

} // namespace merian
