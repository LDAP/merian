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

constexpr std::array<const char*, PATH_DEBUG_MAP_CHANNEL_COUNT> MAP_CHANNEL_DESCRIPTIONS = {
    "Sampled directions per steradian at 'bounce', over the paths of 'paths from'. Without a "
    "pick these are every pixel's paths, binned in the world frame.",
    "Radiance that came back along each direction, per steradian.",
    "Mean color of the radiance that came back along each direction.",
    "Mean luminance of the radiance that came back along each direction.",
    "The material's pdf at the picked surface for the incident direction, evaluated per bin. "
    "Needs a pick.",
    "Sampled density over the pdf reference: white at 1, blue undersampled, red oversampled. "
    "Needs the picked pixel at bounce 0.",
    "Poisson deviation of each bin from the pdf reference; chi2/dof sums them. Needs the picked "
    "pixel at bounce 0.",
    "The BSDF sampled on the GPU at the picked surface, the renderer out of the loop (check BSDF). "
    "Needs a pick.",
    "f * cos per direction, normalized: the distribution an ideal sampler follows. Needs a pick."};

} // namespace

PathDebugNode::NodeStatusFlags PathDebugNode::properties(Properties& config) {
    const std::scoped_lock lock(stats_mutex);
    if (config.is_ui() && frames_since_window < UI_IDLE_FRAMES) {
        config.output_text("The controls live in the Path Debugger window.");
        return {};
    }

    properties_selection(config);
    properties_overlay(config);
    properties_backdrop(config);
    properties_maps(config);
    properties_heat(config);
    properties_pixel_stats(config);
    properties_capture(config);
    properties_export(config);
    return std::exchange(pending_reconnect, false) ? NodeStatusFlags{NEEDS_RECONNECT}
                                                   : NodeStatusFlags{};
}

void PathDebugNode::draw_pick_hint(Properties& config) const {
    if (!picked) {
        config.output_text("Select with ALT+Click");
    } else if (picked_offscreen) {
        config.output_text("the pick is offscreen");
    } else {
        config.output_text("pixel ({}, {})", picked_pixel.x, picked_pixel.y);
    }
}

void PathDebugNode::properties_selection(Properties& config) {
    if (!config.st_begin_child("selection", "selection", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        return;
    }
    bool changed = false;
    draw_pick_hint(config);
    if (config.config_bool("pixel picked", picked, "Unset or press escape to clear the pick.")) {
        if (picked) {
            picked_in_records = picked_pixel;
            changed = true;
        } else {
            clear_selection();
        }
    }
    if (picked && config.config_vec("pixel", picked_pixel)) {
        picked_in_records = picked_pixel;
        picked_offscreen = false;
        picked_pos_valid = false;
        picked_radius = 0.f;
        pending_pick = false;
        changed = true;
    }
    if (config.config_bool("reproject", reproject,
                           "The pick keeps its surface point and follows it as the camera moves; "
                           "kept paths stay. Off, the pick stays on its pixel and a camera move "
                           "clears the kept paths.")) {
        picked_offscreen = false;
    }
    changed |= config.config_options(
        "restrict to", restrict_mode, options(RESTRICT_NAMES), Properties::OptionsStyle::COMBO,
        "Which paths the pick keeps. A region is a sphere around the picked surface point.");
    if (picked_pos_valid && restrict_mode >= static_cast<int32_t>(PATH_DEBUG_RESTRICT_THROUGH)) {
        changed |= config.config_vec("region centre", picked_pos, "World space.");
        changed |= config.config_float("region radius", picked_radius,
                                       "World units; ctrl+scroll adjusts it.", 0.001f);
        picked_radius = std::max(picked_radius, 0.f);
    }

    config.st_separate("expressions");
    std::string text_a = expr_a;
    if (config.config_text(
            "expression a", text_a, false,
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
    if (config.config_text("expression b", text_b, false, "Same syntax, for selection B.")) {
        set_expression(true, text_b);
    }
    if (!compiled_b.valid) {
        config.output_text("invalid expression");
    }

    config.st_separate("constraints");
    changed |= config.config_options("contribution", contribution_filter, options(FINITE_NAMES),
                                     Properties::OptionsStyle::COMBO,
                                     "Keep only finite or only non-finite paths.");
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
            "Most recent: the last matching paths over frames, a uniform share of a frame with "
            "more. "
            "Brightest: the brightest paths since the last change. Brightest fraction: that share "
            "of the frame's brightest paths. Resampled: a sample without replacement, proportional "
            "to contribution, over the recent history.")) {
        kept_dirty = true;
        isolate_path = -1;
    }
    kept_dirty |= config.config_int(
        "max paths", overlay_max_paths,
        overlay_kept() ? "Brightest, resampled and most recent keep at most 64." : "", 1,
        static_cast<int32_t>(overlay_kept() ? PATH_DEBUG_KEPT_SLOTS : MAX_DRAW));
    if (overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_FIREFLIES)) {
        config.config_float("brightest fraction", overlay_brightest_fraction, "", 1e-5f);
    }
    if (overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_RESAMPLED)) {
        config.config_float(
            "history", resample_history,
            "Time constant in frames over which kept paths lose weight; 0 weighs every path "
            "since the last change alike.",
            1.f);
        resample_history = std::max(resample_history, 0.f);
    }
    config.config_int(
        "isolate path", isolate_path,
        overlay_kept() ? "The inspector's path number; the kept paths stay while one is "
                         "isolated. -1 = all."
                       : "Record index; -1 = all.",
        -1,
        static_cast<int32_t>(overlay_kept() ? PATH_DEBUG_KEPT_SLOTS : record_path_capacity) - 1);

    config.st_separate("style");
    config.config_options("color", overlay_color, options(COLOR_NAMES),
                          Properties::OptionsStyle::COMBO);
    if (overlay_color == static_cast<int32_t>(PATH_DEBUG_COLOR_LUMINANCE)) {
        config.config_float("color scale", params.overlay_color_scale, "", 0.01f);
    }
    config.config_options("depth test", overlay_depth_mode, options(DEPTH_NAMES),
                          Properties::OptionsStyle::COMBO,
                          gbuffer_connected ? "Against the primary surfaces of the gbuffer."
                                            : "Needs the gbuffer input.");
    config.config_float("thickness", params.overlay_thickness,
                        "Width in pixels at the path's first vertex.", 0.1f);
    params.overlay_thickness = std::clamp(params.overlay_thickness, 1.f, 8.f);
    config.config_percent("alpha", params.overlay_alpha);
    config.st_end_child();
}

void PathDebugNode::properties_backdrop(Properties& config) {
    if (!config.st_begin_child("backdrop", "backdrop", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        return;
    }
    config.config_options("show", backdrop, options(BACKDROP_NAMES),
                          Properties::OptionsStyle::COMBO, "What the paths are drawn on.");
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
    if (config.config_path("reference image", reference_path)) {
        reference_dirty = true;
    }
    if (!reference_path.empty() && !reference_loaded) {
        config.output_text("could not load the reference");
    }
    config.config_bool("flip render/reference", reference_flip);
    if (reference_flip) {
        config.config_float("flip rate", reference_flip_hz, "In Hz.", 0.1f);
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
    maps_dirty |= config.config_options("draws", map_draws, options(DRAWS_NAMES),
                                        Properties::OptionsStyle::COMBO,
                                        "Which technique's directions are binned.");
    maps_dirty |= config.config_int("bounce", map_bounce, "-1 = all", -1, 30);
    maps_dirty |= config.config_options("frame", map_frame, options(FRAME_NAMES),
                                        Properties::OptionsStyle::COMBO,
                                        "The local frame is the picked surface's; without a "
                                        "visible pick the maps bin in the world frame.");
    maps_dirty |= config.config_options("transform", map_transform, options(TRANSFORM_NAMES),
                                        Properties::OptionsStyle::COMBO);
    if (config.config_int("log2 resolution", map_res_log2, "3 = 8x8 ... 10 = 1024x1024", 3, 10)) {
        maps_dirty = true;
        pending_reconnect = true;
    }
    maps_dirty |= config.config_options(
        "pdf reference", map_pdf_ref, options(PDF_REF_NAMES), Properties::OptionsStyle::COMBO,
        "What density / pdf and z-score test against. BSDF pdf: the material's pdf at the picked "
        "surface. recorded pdf: the pdf each sampler recorded with its own sample, for guided and "
        "resampled samplers too. Needs the picked pixel at bounce 0.");
    maps_dirty |= config.config_options("incident direction", wi_mode, options(WI_NAMES),
                                        Properties::OptionsStyle::COMBO,
                                        "Where the BSDF references are evaluated from.");
    if (wi_mode == static_cast<int32_t>(PATH_DEBUG_WI_MANUAL)) {
        maps_dirty |= config.config_float("polar angle", wi_theta_deg, "In degrees.", 1.f);
        maps_dirty |= config.config_float("azimuth", wi_phi_deg, "In degrees.", 1.f);
        wi_theta_deg = std::clamp(wi_theta_deg, 0.f, 179.f);
        wi_phi_deg = std::clamp(wi_phi_deg, -180.f, 180.f);
    }
    maps_dirty |=
        config.config_int("batch frames", map_batch_frames,
                          "Frames per batch for the variance of reused samples.", 1, 4096);
    config.config_bool("accumulate", map_accumulate);
    config.config_int("analysis paths per frame", analysis_paths, "0 = all", 0, 1 << 22);
    if (config.config_bool("reset")) {
        maps_dirty = true;
    }

    config.st_separate("panels");
    config.config_bool("show panels", panels_enabled,
                       "Over the right edge of the window; scroll over them to page.");
    for (uint32_t channel = 0; channel < MAP_CHANNEL_COUNT; channel++) {
        config.config_bool(MAP_CHANNEL_NAMES[channel], panel_enabled[channel],
                           MAP_CHANNEL_DESCRIPTIONS[channel]);
    }
    config.config_int("panel size", panel_size, "In pixels at a display scale of 1.", 64, 1024);
    config.config_float("exposure", params.map_exposure, "", 0.01f);
    std::vector<std::string> sphere_options{"off"};
    sphere_options.insert(sphere_options.end(), MAP_CHANNEL_NAMES.begin(), MAP_CHANNEL_NAMES.end());
    int32_t sphere_option = sphere_channel + 1;
    config.config_options("sphere", sphere_option, sphere_options, Properties::OptionsStyle::COMBO,
                          "A channel drawn on a sphere at the picked surface.");
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
    if (config.config_int("log2 grid size", grid_slots_log2, "", 8, 26)) {
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
    if (!config.st_begin_child("pixel stats", "per-pixel statistics",
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
    config.config_bool("freeze", freeze,
                       "Holds the recorded paths; everything keeps analyzing and drawing them.");
    config.config_bool("auto subsample", auto_keep_prob,
                       "Fits the keep probability to the record buffer.");
    if (!auto_keep_prob) {
        config.config_percent("keep probability", keep_prob);
        keep_prob = std::clamp(keep_prob, MIN_KEEP_PROB, 1.f);
    }
    config.config_bool("picked pixel only", record_picked_pixel_only,
                       "Records only the picked pixel, at every sample.");
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
