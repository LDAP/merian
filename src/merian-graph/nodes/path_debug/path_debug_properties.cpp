#include "merian-graph/nodes/path_debug/path_debug.hpp"

#include "merian-shaders/debug/path_record.hpp"
#include "merian/vk/utils/image_export.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace merian {

namespace {

template <std::size_t N> std::vector<std::string> options(const std::array<const char*, N>& names) {
    return {names.begin(), names.end()};
}

} // namespace

PathDebugNode::NodeStatusFlags PathDebugNode::properties(Properties& config) {
    NodeStatusFlags flags{};
    const std::scoped_lock lock(stats_mutex);

    if (config.config_text("imgui event pattern", imgui_event_pattern, true,
                           "Graph event that carries the ImGui frame.")) {
        flags |= NEEDS_RECONNECT;
    }
    if (config.is_ui()) {
        config.output_text("The controls live in the Path Debugger window.");
        return flags;
    }

    properties_selection(config);
    properties_overlay(config);
    properties_backdrop(config);
    properties_maps(config);
    properties_heat(config);
    properties_pixel_stats(config);
    properties_capture(config);
    properties_export(config);
    if (std::exchange(pending_reconnect, false)) {
        flags |= NEEDS_RECONNECT;
    }
    return flags;
}

void PathDebugNode::properties_selection(Properties& config) {
    if (!config.st_begin_child("selection", "selection", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        return;
    }
    bool changed = false;
    changed |=
        config.config_options("restrict to", restrict_mode, options(RESTRICT_NAMES),
                              Properties::OptionsStyle::COMBO, "Which paths the pick keeps.");
    changed |= config.config_bool("pixel picked", picked);
    if (picked && config.config_vec("pixel", picked_pixel)) {
        picked_pos_valid = false;
        picked_radius = 0.f;
        pending_pick = false;
        changed = true;
    }
    if (picked_pos_valid && restrict_mode >= static_cast<int32_t>(PATH_DEBUG_RESTRICT_THROUGH)) {
        changed |= config.config_vec("region centre", picked_pos, "World space.");
        changed |= config.config_float("region radius", picked_radius,
                                       "World units; ctrl+scroll adjusts it.", 0.001f);
        picked_radius = std::max(picked_radius, 0.f);
    }
    changed |= config.config_options("contribution", contribution_filter, options(FINITE_NAMES),
                                     Properties::OptionsStyle::COMBO,
                                     "Keep only finite or only non-finite paths.");

    config.st_separate("expressions");
    std::string text_a = expr_a;
    if (config.config_text(
            "expression A", text_a, false,
            "Per scatter event: S G D U specular, glossy, diffuse, unclassified; R T reflected, "
            "transmitted; . any; <RD> both; [SG] either; repeats * + ? {n} {n,m}. Matches the "
            "whole path; a trailing N instead ends at a vertex that connected to a light by NEE "
            "(.N direct, .*S.+N after a specular event).")) {
        set_expression(false, text_a);
    }
    if (!compiled_a.valid) {
        config.output_text("invalid expression");
    }
    std::string text_b = expr_b;
    if (config.config_text("expression B", text_b, false)) {
        set_expression(true, text_b);
    }
    if (!compiled_b.valid) {
        config.output_text("invalid expression");
    }

    config.st_separate("constraints");
    changed |= config.config_vec("scatter events", scatter_range, "Min and max.");
    changed |= config.config_int("material", material_id, "Material id; -1 = any.", -1,
                                 static_cast<int32_t>(PATH_RECORD_MATERIAL_NONE) - 1);
    if (config.st_begin_child("method", "sampling method")) {
        for (uint32_t m = 0; m < PATH_RECORD_METHOD_NAMES.size(); m++) {
            bool set = (method_mask & (1u << m)) != 0;
            if (config.config_bool(PATH_RECORD_METHOD_NAMES[m], set)) {
                method_mask = set ? (method_mask | (1u << m)) : (method_mask & ~(1u << m));
                changed = true;
            }
        }
        changed |= config.config_options("mode", method_mode, options(METHOD_MODE_NAMES),
                                         Properties::OptionsStyle::COMBO,
                                         "Match on any or on every scatter vertex.");
        config.st_end_child();
    }
    if (changed) {
        invalidate_selection();
    }
    config.st_end_child();
}

void PathDebugNode::properties_overlay(Properties& config) {
    if (!config.st_begin_child("overlay", "overlay", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        return;
    }
    kept_dirty |= config.config_options("paths from", overlay_source, options(SOURCE_NAMES),
                                        Properties::OptionsStyle::COMBO);
    if (config.config_options(
            "keep", overlay_rank, options(RANK_NAMES), Properties::OptionsStyle::COMBO,
            "Resampled: paths drawn proportional to their contribution from every "
            "path since the last change. Most recent: the last matching paths over frames.")) {
        kept_dirty = true;
        isolate_path = -1;
    }
    kept_dirty |=
        config.config_int("max paths", overlay_max_paths, "", 1,
                          static_cast<int32_t>(overlay_kept() ? PATH_DEBUG_KEPT_SLOTS : MAX_DRAW));
    if (overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_FIREFLIES)) {
        config.config_float("brightest fraction", overlay_brightest_fraction, "", 1e-5f);
    }
    config.config_options("depth test", overlay_depth_mode, options(DEPTH_NAMES),
                          Properties::OptionsStyle::COMBO,
                          gbuffer_connected ? "" : "Needs the gbuffer input.");
    config.config_options("color", overlay_color, options(COLOR_NAMES),
                          Properties::OptionsStyle::COMBO);
    if (overlay_color == static_cast<int32_t>(PATH_DEBUG_COLOR_LUMINANCE)) {
        config.config_float("color scale", params.overlay_color_scale, "", 0.01f);
    }
    config.config_float("thickness", params.overlay_thickness,
                        "Width in pixels at the path's first vertex.", 0.1f);
    params.overlay_thickness = std::clamp(params.overlay_thickness, 1.f, 8.f);
    config.config_percent("alpha", params.overlay_alpha);
    config.config_int("isolate path", isolate_path, "Record index; -1 = all.", -1,
                      static_cast<int32_t>(record_path_capacity));
    config.st_end_child();
}

void PathDebugNode::properties_backdrop(Properties& config) {
    if (!config.st_begin_child("backdrop", "backdrop", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        return;
    }
    config.config_options("show", backdrop, options(BACKDROP_NAMES),
                          Properties::OptionsStyle::COMBO);
    const bool shows_filtered =
        backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_FILTERED_A) ||
        backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_FILTERED_B) ||
        backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_SPLIT_FILTERED);
    if (shows_filtered) {
        config.config_float("exposure", params.filtered_exposure, "0 = raw radiance.", 0.01f);
    }
    if (backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_DIFFERENCE)) {
        config.config_float("difference gain", params.difference_gain, "", 0.1f);
    }
    if (backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_SPLIT_REFERENCE) ||
        backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_SPLIT_FILTERED)) {
        config.config_percent("split", params.backdrop_split);
    }
    if (config.config_text("reference image", reference_path, true, "")) {
        reference_dirty = true;
    }
    if (!reference_path.empty()) {
        if (!reference_loaded) {
            config.output_text("could not load the reference");
        }
    }
    config.config_bool("flip render/reference", reference_flip);
    if (reference_flip) {
        config.config_float("flip rate (Hz)", reference_flip_hz, "", 0.1f);
    }
    config.st_end_child();
}

void PathDebugNode::properties_maps(Properties& config) {
    if (!config.st_begin_child("maps", "directional maps",
                               Properties::ChildFlagBits::DEFAULT_OPEN)) {
        return;
    }
    if (!gbuffer_connected) {
        config.output_text("needs the gbuffer input");
    }
    maps_dirty |= config.config_options("paths from", map_source, options(SOURCE_NAMES),
                                        Properties::OptionsStyle::COMBO);
    if (config.config_int("resolution (log2)", map_res_log2, "3 = 8x8 ... 10 = 1024x1024", 3, 10)) {
        maps_dirty = true;
        pending_reconnect = true;
    }
    maps_dirty |= config.config_options("transform", map_transform, options(TRANSFORM_NAMES),
                                        Properties::OptionsStyle::COMBO);
    maps_dirty |= config.config_options("frame", map_frame, options(FRAME_NAMES),
                                        Properties::OptionsStyle::COMBO);
    maps_dirty |= config.config_int("bounce", map_bounce, "-1 = all", -1, 30);
    maps_dirty |= config.config_options("draws", map_draws, options(DRAWS_NAMES),
                                        Properties::OptionsStyle::COMBO);
    maps_dirty |=
        config.config_int("batch frames", map_batch_frames,
                          "Frames per batch for the variance of reused samples.", 1, 4096);
    maps_dirty |= config.config_options("pdf reference", map_pdf_ref, options(PDF_REF_NAMES),
                                        Properties::OptionsStyle::COMBO,
                                        "Needs the picked pixel at bounce 0.");
    if (!config.is_ui()) {
        maps_dirty |= config.config_options("incident direction", wi_mode, options(WI_NAMES),
                                            Properties::OptionsStyle::COMBO);
        maps_dirty |= config.config_float("polar angle (deg)", wi_theta_deg, "", 1.f);
        maps_dirty |= config.config_float("azimuth (deg)", wi_phi_deg, "", 1.f);
        wi_theta_deg = std::clamp(wi_theta_deg, 0.f, 179.f);
        wi_phi_deg = std::clamp(wi_phi_deg, -180.f, 180.f);
    }
    config.config_bool("accumulate", map_accumulate);
    config.config_int("analysis paths / frame", analysis_paths, "0 = all", 0, 1 << 22);
    if (config.config_bool("reset")) {
        maps_dirty = true;
    }
    config.st_separate("panels");
    config.config_bool("show panels", panels_enabled);
    for (uint32_t channel = 0; channel < MAP_CHANNEL_COUNT; channel++) {
        config.config_bool(MAP_CHANNEL_NAMES[channel], panel_enabled[channel]);
    }
    auto panel_size = static_cast<int32_t>(params.map_panel_size);
    config.config_int("panel size", panel_size, "", 64, 1024);
    params.map_panel_size = static_cast<uint32_t>(panel_size);
    config.config_float("exposure", params.map_exposure, "", 0.01f);
    std::vector<std::string> sphere_options{"off"};
    sphere_options.insert(sphere_options.end(), MAP_CHANNEL_NAMES.begin(), MAP_CHANNEL_NAMES.end());
    int32_t sphere_option = sphere_channel + 1;
    config.config_options("sphere", sphere_option, sphere_options, Properties::OptionsStyle::COMBO);
    sphere_channel = sphere_option - 1;
    config.config_percent("sphere size", params.sphere_radius);
    config.st_end_child();
}

void PathDebugNode::properties_heat(Properties& config) {
    if (!config.st_begin_child("heat", "scene heatmap", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        return;
    }
    if (!gbuffer_connected) {
        config.output_text("needs the gbuffer input");
    }
    heat_dirty |= config.config_options("mode", heat_mode, options(HEAT_NAMES),
                                        Properties::OptionsStyle::COMBO);
    heat_dirty |= config.config_options("paths from", heat_source, options(SOURCE_NAMES),
                                        Properties::OptionsStyle::COMBO);
    heat_dirty |= config.config_vec("bounce range", heat_bounce_range);
    heat_dirty |= config.config_float("cell size", params.heat_cell_size, "World units.", 0.001f);
    params.heat_cell_size = std::max(params.heat_cell_size, 1e-4f);
    if (config.config_int("grid size (log2)", grid_slots_log2, "", 8, 26)) {
        heat_dirty = true;
        pending_reconnect = true;
    }
    config.config_percent("alpha", heat_alpha, "0 = keep forever.");
    config.config_float("exposure", params.heat_exposure, "", 0.01f);
    config.config_percent("opacity", params.heat_opacity);
    config.config_bool("smooth", params.heat_smooth);
    if (config.config_bool("clear")) {
        heat_dirty = true;
    }
    config.st_end_child();
}

void PathDebugNode::properties_pixel_stats(Properties& config) {
    if (!config.st_begin_child("pixel_stats", "per-pixel statistics",
                               Properties::ChildFlagBits::DEFAULT_OPEN)) {
        return;
    }
    config.config_options("overlay", pixel_stats_view, options(PIXEL_STAT_NAMES),
                          Properties::OptionsStyle::COMBO);
    if (pixel_stats_view != static_cast<int32_t>(PATH_DEBUG_PIXEL_STAT_OFF)) {
        config.config_float("scale", params.pixel_stats_scale, "Colormap maximum.", 0.1f);
    }
    if (config.config_options("paths from", pixel_stats_source, options(SOURCE_NAMES),
                              Properties::OptionsStyle::COMBO)) {
        pixel_stats_dirty = true;
    }
    if (config.config_bool("reset")) {
        pixel_stats_dirty = true;
    }
    config.st_end_child();
}

void PathDebugNode::properties_capture(Properties& config) {
    if (!config.st_begin_child("capture", "capture", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        return;
    }
    config.config_bool("freeze", freeze);
    config.config_bool("auto subsample", auto_keep_prob,
                       "Fits the keep probability to the record buffer.");
    if (!auto_keep_prob) {
        config.config_percent("keep probability", keep_prob);
        keep_prob = std::clamp(keep_prob, MIN_KEEP_PROB, 1.f);
    }
    config.config_bool("picked pixel only", record_picked_pixel_only);
    config.st_end_child();
}

void PathDebugNode::properties_export(Properties& config) {
    if (!config.st_begin_child("export", "export", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        return;
    }
    std::ignore = config.config_text("filename", export_path, false, "Without extension.");
    config.config_options("format", export_format, image_export_format_names(),
                          Properties::OptionsStyle::COMBO);
    config.config_bool("keep alpha", export_keep_alpha);
    config.config_bool("embed metadata", export_metadata,
                       "The merian version and the graph config.");
    export_next |= config.config_bool("export now");
    config.st_end_child();
}

} // namespace merian
