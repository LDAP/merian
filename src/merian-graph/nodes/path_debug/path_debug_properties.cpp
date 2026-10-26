#include "merian-graph/nodes/path_debug/path_debug.hpp"

#include "merian-shaders/debug/path_record.hpp"
#include "merian/vk/utils/image_export.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace merian {

namespace {

template <std::size_t N> std::vector<std::string> options(const std::array<const char*, N>& names) {
    return {names.begin(), names.end()};
}

constexpr int32_t ANCHOR_SURFACE_POINT = 0;
constexpr int32_t ANCHOR_PIXEL = 1;
constexpr std::array<const char*, 2> ANCHOR_NAMES = {"surface point", "pixel"};
constexpr std::array<const char*, 3> FINITE_NAMES = {"any", "finite", "NaN or Inf"};
constexpr std::array<const char*, 2> METHOD_MODE_NAMES = {"any event", "every event"};
constexpr std::array<const char*, 4> RANK_NAMES = {"most recent", "brightest", "brightest share",
                                                   "resampled by contribution"};
constexpr std::array<const char*, 3> DEPTH_NAMES = {"x-ray", "hide", "dash"};
constexpr std::array<const char*, 3> TRANSFORM_NAMES = {"octahedral", "lat-long",
                                                        "hemisphere (equal-area)"};
constexpr std::array<const char*, 2> FRAME_NAMES = {"world (y up)", "local shading"};
// without the off state at 0
constexpr std::array<const char*, 8> HEAT_NAMES = {
    "vertex count", "contribution", "throughput", "incident radiance",
    "BSDF·cos/pdf", "NEE success",  "NEE failed", "path ends"};
constexpr std::array<const char*, 8> HEAT_HELP = {
    "Per cell, the vertices of the selection's paths.",
    "Per cell, the light that reaches the camera through the vertices.",
    "Mean throughput T, the product of BSDF·cos/pdf from the camera: how much light arriving at "
    "a vertex reaches the camera.",
    "Mean radiance arriving along the drawn direction: the contribution after a vertex over T.",
    "Mean BSDF·cos/pdf of the drawn directions, the BSDF for the incoming and the drawn "
    "direction: above 1 a draw was rarer than the integrand asks for.",
    "Per cell, the NEE connections that delivered light.",
    "Per cell, the NEE connections that delivered nothing: occluded, or the light behind the "
    "surface.",
    "Per cell, the vertices where paths stop: roulette, the length limit, a refused sample."};
constexpr std::array<const char*, 7> PIXEL_STAT_NAMES = {
    "mean luminance",          "sample count", "variance",     "relative variance",
    "relative standard error", "max / mean",   "zero fraction"};
constexpr std::array<const char*, 7> PIXEL_STAT_HELP = {
    "mean(X), X what one recorded path adds to the pixel through the selection.",
    "n, the recorded paths of the pixel.",
    "Var(X) = mean(X²) - mean(X)².",
    "Var(X) / mean(X)²: one path's noise, whatever the brightness.",
    "sqrt(Var(X) / n) / mean(X): the error of the pixel's mean, falls with n.",
    "max(X) / mean(X): fireflies lie far above 1.",
    "(paths with X = 0) / n."};

constexpr const char* EXPRESSION_HELP =
    "Matches the scatter events of a light path, camera to light. Empty matches every path.\n"
    "\n"
    "Events\n"
    "  S G D U: specular, glossy, diffuse, unclassified\n"
    "  R T: reflection, transmission\n"
    "  .: any\n"
    "Techniques: how the event's outgoing direction was drawn\n"
    "  'b' BSDF, 'g' guiding, 'n' NEE, 'c' cosine, 'u' uniform\n"
    "  'rrb' replayed BSDF, 'rrn' replayed NEE, 'rec' reconnection\n"
    "  The last event's technique is how the light was reached.\n"
    "Combining\n"
    "  <D'g'>: all of, [SG]: either\n"
    "  * + ? {n} {n,m} {n,}: repeats\n"
    "  L alone: a light the camera sees\n"
    "Examples\n"
    "  .: direct light\n"
    "  .'n': direct light by NEE\n"
    "  .'b': direct light a BSDF sample hit\n"
    "  .*<T>.*: refracted somewhere";

} // namespace

PathDebugNode::NodeStatusFlags PathDebugNode::properties(Properties& config) {
    if (config.is_ui() && frames_since_window < UI_IDLE_FRAMES) {
        config.output_text("The controls live in the Path Debugger window.");
        return {};
    }

    const auto section = [&](const char* id, const char* label, const Properties::ChildFlags flags,
                             const auto& content) {
        if (config.st_begin_child(id, label, flags)) {
            content();
            config.st_end_child();
        }
    };
    constexpr Properties::ChildFlags OPEN = Properties::ChildFlagBits::DEFAULT_OPEN;
    section("selection", "selection", OPEN, [&] {
        properties_selection(config);
        properties_selection_details(config);
    });
    section("paths", "paths", OPEN, [&] {
        properties_color(config);
        properties_path_choice(config);
        properties_line_style(config);
    });
    section("filter", "filter", OPEN, [&] {
        properties_filter(config);
        properties_techniques(config);
    });
    section("view", "view", OPEN, [&] {
        properties_image(config);
        section("overlay", "overlay", OPEN, [&] {
            config.config_options("type", false_color, options(FALSE_COLOR_NAMES),
                                  Properties::OptionsStyle::RADIO_BUTTON,
                                  "World space: per cell of the scene. Screen space: per pixel.");
            properties_false_color(config);
        });
        section("sphere", "sphere", OPEN, [&] { properties_sphere(config); });
        section("reference", "reference", OPEN, [&] { properties_reference(config); });
    });
    section("analysis", "analysis", OPEN, [&] {
        section("bsdf check", "BSDF check", {}, [&] { properties_bsdf_check(config); });
        section("panels", "panels", {}, [&] {
            properties_panels(config);
            properties_bins(config);
        });
        section("sampling", "sampling", {}, [&] { properties_sampling(config); });
    });
    section("capture", "capture", OPEN, [&] { properties_capture(config); });
    section("export", "export", OPEN, [&] { properties_export(config); });
    section("preferences", "preferences", OPEN, [&] { properties_keys(config); });
    return std::exchange(pending_reconnect, false) ? NodeStatusFlags{NEEDS_RECONNECT}
                                                   : NodeStatusFlags{};
}

void PathDebugNode::properties_selection(Properties& config) {
    config.output_text(selection_status());
    int32_t choice = selection_choice();
    if (config.config_options("select", choice, options(SELECTION_NAMES),
                              Properties::OptionsStyle::RADIO_BUTTON,
                              "Point: the pixel's paths that start at the clicked surface. The "
                              "others: the light paths that hit, start or end on a sphere around "
                              "the click or on what the click sees; the sky selects the "
                              "environment.")) {
        set_selection_choice(choice);
    }
    if (select_kind == SELECT_REGION) {
        int32_t match = region_match;
        if (config.config_options("match", match, options(MATCH_NAMES),
                                  Properties::OptionsStyle::RADIO_BUTTON,
                                  "Light paths, camera to light, that hit the target, start or end "
                                  "on it.")) {
            set_region_match(match);
        }
    }
    if (has_selection &&
        config.config_bool("clear", fmt::format("{}+Backspace does the same.", modifier_name()))) {
        clear_selection();
    }
}

void PathDebugNode::properties_selection_details(Properties& config) {
    bool changed = false;
    if (targets_ids()) {
        RegionTarget ids = target;
        bool set = config.config_bool("environment", ids.environment);
        set |= config.config_int("mesh", ids.mesh, "Id; -1 = none.");
        set |= config.config_int("node", ids.node, "Scene node id; -1 = none.");
        set |= config.config_int("triangle", ids.primitive, "Of the mesh.");
        set |= config.config_int("material", ids.material, "Id; -1 = none.");
        if (set) {
            set_target(ids);
        }
    }
    if (select_kind == SELECT_REGION && region_target == TARGET_SPHERE && selected_pos_valid) {
        changed |= config.config_float(
            "radius", selected_radius,
            fmt::format("World units; {}+scroll over the image.", modifier_name()), 0.001f);
        selected_radius = std::max(selected_radius, 0.f);
    }
    int32_t anchor = reproject ? ANCHOR_SURFACE_POINT : ANCHOR_PIXEL;
    if (config.config_options("anchor", anchor, options(ANCHOR_NAMES),
                              Properties::OptionsStyle::COMBO,
                              "Surface point: follows the surface when the camera moves. Pixel: "
                              "stays on the pixel.")) {
        reproject = anchor == ANCHOR_SURFACE_POINT;
        selected_offscreen = false;
        changed = true;
    }
    int2 pixel = has_selection ? selected_pixel : int2(-1);
    if (config.config_vec("pixel", pixel,
                          "In the pixels the paths are traced in; -1 = no selection.")) {
        if (pixel.x >= 0 && pixel.y >= 0) {
            select_at(pixel);
        } else {
            clear_selection();
        }
    }
    if (has_selection && selected_pos_valid) {
        changed |= config.config_vec("surface point", selected_pos, "World space.");
    }
    if (changed) {
        invalidate_selection();
    }
}

void PathDebugNode::properties_color(Properties& config) {
    config.config_options("color", overlay_color, options(COLOR_NAMES),
                          Properties::OptionsStyle::COMBO,
                          "Path luminance: what the selection's light paths add to the pixel. "
                          "Transported radiance: the radiance each segment carries towards the "
                          "camera; transported color its hue, the light's color after every "
                          "surface it left. BSDF·cos/pdf: the weight a segment's direction was "
                          "drawn with. Off: no paths.");
}

void PathDebugNode::properties_filter(Properties& config) {
    std::string text = expression;
    if (config.config_text("expression", text, false, EXPRESSION_HELP)) {
        set_expression(text);
    }
    if (!compiled_expression.valid) {
        config.output_text("invalid expression");
    }

    bool changed =
        config.config_options("contribution", contribution_filter, options(FINITE_NAMES),
                              Properties::OptionsStyle::COMBO, "Finite or non-finite paths only.");
    changed |= config.config_vec("scatter events", scatter_range,
                                 fmt::format("Min and max; {} = no limit.", SCATTER_UNLIMITED));
    scatter_range.x = std::clamp(scatter_range.x, 0, SCATTER_UNLIMITED);
    scatter_range.y = std::clamp(scatter_range.y, scatter_range.x, SCATTER_UNLIMITED);
    if (changed) {
        invalidate_filter();
    }
}

void PathDebugNode::properties_techniques(Properties& config) {
    config.st_separate("technique");
    for (uint32_t m = 0; m < PATH_RECORD_METHOD_NAMES.size(); m++) {
        bool set = (method_mask & (1u << m)) != 0;
        if (config.config_bool(PATH_RECORD_METHOD_NAMES[m], set)) {
            set_method(m, set);
        }
    }
    properties_technique_match(config);
}

void PathDebugNode::properties_technique_match(Properties& config) {
    if (config.config_options("match", method_mode, options(METHOD_MODE_NAMES),
                              Properties::OptionsStyle::COMBO,
                              "None ticked matches any technique.")) {
        invalidate_filter();
    }
}

void PathDebugNode::properties_path_choice(Properties& config) {
    if (config.config_options(
            "draw", overlay_rank, options(RANK_NAMES), Properties::OptionsStyle::COMBO,
            "Most recent: the latest paths. Brightest: the brightest since the last change. "
            "Brightest share: that share of each frame's brightest paths. Resampled: proportional "
            "to contribution over the recent history.")) {
        kept_dirty = true;
        isolate_path.reset();
    }
    kept_dirty |= config.config_int("max paths", overlay_max_paths, "", 0,
                                    static_cast<int32_t>(PATH_DEBUG_KEPT_SLOTS));
    overlay_max_paths =
        std::clamp(overlay_max_paths, 0, static_cast<int32_t>(PATH_DEBUG_KEPT_SLOTS));
    if (overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_BRIGHTEST_SHARE)) {
        config.config_float("share", overlay_brightest_share, "Of the frame's paths.", 1e-5f);
        overlay_brightest_share = std::clamp(overlay_brightest_share, 1e-7f, 1.f);
    }
    if (overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_RESAMPLED)) {
        config.config_float("fade frames", resample_history,
                            "A drawn path's weight falls to 1/e in this many frames; 0 keeps it "
                            "until a change.",
                            1.f);
        resample_history = std::max(resample_history, 0.f);
    }
    if (config.config_bool("filtered", paths_filtered,
                           "The drawn paths and the list pass the filter.")) {
        kept_dirty = true;
        isolate_path.reset();
    }
}

void PathDebugNode::properties_line_style(Properties& config) {
    if (overlay_rank != static_cast<int32_t>(PATH_DEBUG_RANK_BRIGHTEST_SHARE)) {
        config.config_bool("keep on camera move", keep_on_camera_move,
                           "Off, a camera move starts the drawn paths over.");
    }
    config.config_options("behind geometry", overlay_depth_mode, options(DEPTH_NAMES),
                          Properties::OptionsStyle::COMBO,
                          gbuffer_connected ? "Segments behind the primary surfaces of the gbuffer."
                                            : "gbuffer required");
    config.config_float("thickness", overlay_thickness, "Pixels at the path's first vertex.", 0.1f);
    overlay_thickness = std::clamp(overlay_thickness, 1.f, 8.f);
    config.config_percent("opacity", overlay_opacity);
    if (overlay_color == static_cast<int32_t>(PATH_DEBUG_COLOR_LUMINANCE) ||
        overlay_color == static_cast<int32_t>(PATH_DEBUG_COLOR_CONTRIBUTION)) {
        config.config_float("exposure", overlay_exposure,
                            "Scales the value before the logarithmic color map.", 0.01f);
    }
}

void PathDebugNode::properties_image(Properties& config) {
    config.config_options("image", backdrop, options(BACKDROP_NAMES),
                          Properties::OptionsStyle::COMBO,
                          "What the paths are drawn on. Current frame and accumulated: the "
                          "recorded paths; filtered, what the region and the filter add.");
    if (config.is_ui() && backdrop_missing) {
        config.output_text("{} required · showing source input", missing_image_input());
    }
    config.config_float("exposure", image_exposure,
                        "Scales the image; the recorded radiance is tone mapped.", 0.01f);
    image_exposure = std::max(image_exposure, 0.f);
    config.config_options("background", background, options(BACKGROUND_NAMES),
                          Properties::OptionsStyle::COMBO,
                          "Once a frozen camera moves: wireframe in the gaps and over nearer "
                          "geometry, shaded albedo in the gaps and over hidden images.");
    if (config.is_ui() && !gbuffer_connected &&
        background != static_cast<int32_t>(PATH_DEBUG_BACKGROUND_OFF) &&
        path_debug_backdrop_recorded(static_cast<uint32_t>(backdrop))) {
        config.output_text("gbuffer required");
    }
    if (path_debug_backdrop_accumulated(static_cast<uint32_t>(backdrop)) &&
        config.config_bool("reset")) {
        image_dirty = true;
    }
}

void PathDebugNode::properties_false_color(Properties& config) {
    if (false_color == FALSE_COLOR_OFF) {
        return;
    }
    if (false_color == FALSE_COLOR_HEAT) {
        if (!gbuffer_connected) {
            config.output_text("gbuffer required");
        }
        heat_dirty |= config.config_options("quantity", heat_quantity, options(HEAT_NAMES),
                                            Properties::OptionsStyle::COMBO,
                                            "Of the selection's paths in world-space cells.");
        if (config.is_ui()) {
            config.output_text(HEAT_HELP.at(heat_quantity));
        }
    } else {
        config.config_options("statistic", pixel_statistic, options(PIXEL_STAT_NAMES),
                              Properties::OptionsStyle::COMBO,
                              "Over the frames, of every path recorded at a pixel.");
        if (config.is_ui()) {
            config.output_text(PIXEL_STAT_HELP.at(pixel_statistic));
        }
    }
    if (config.config_bool("filtered", false_color_filtered,
                           "Only what the region and the filter pass.")) {
        heat_dirty = true;
        pixel_stats_dirty = true;
    }
    config.config_percent("opacity", false_color_opacity);
    if (false_color == FALSE_COLOR_HEAT) {
        heat_dirty |= config.config_vec("scatter events", heat_bounce_range,
                                        "The vertices this far from the camera, min and max.");
        heat_bounce_range.x = std::clamp(heat_bounce_range.x, 0, HEAT_BOUNCE_MAX);
        heat_bounce_range.y = std::clamp(heat_bounce_range.y, heat_bounce_range.x, HEAT_BOUNCE_MAX);
        heat_dirty |= config.config_float("cell size", heat_cell_size, "World units.", 0.001f);
        heat_cell_size = std::max(heat_cell_size, 1e-4f);
        config.config_float("fade frames", heat_history,
                            "A cell's weight falls to 1/e in this many frames; 0 keeps it.", 1.f);
        heat_history = std::max(heat_history, 0.f);
        config.config_float("exposure", heat_exposure, "", 0.01f);
        config.config_bool("blur", heat_smooth, "Averages each cell with its six neighbors.");
        if (config.config_int("log2 cells", grid_slots_log2, "The cells the grid holds.",
                              GRID_SLOTS_LOG2_MIN, GRID_SLOTS_LOG2_MAX)) {
            heat_dirty = true;
            pending_reconnect = true;
        }
        if (config.config_bool("reset")) {
            heat_dirty = true;
        }
        if (config.is_ui() && readback_valid) {
            const uint32_t dropped = latest_readback.frame.grid_dropped;
            config.output_text("grid {:.0f} % full{}",
                               100.f * static_cast<float>(latest_readback.frame.grid_cells) /
                                   static_cast<float>(std::max(params.grid_slots, 1u)),
                               dropped > 0 ? fmt::format(" · {} splats dropped", dropped) : "");
        }
        return;
    }
    config.config_float("exposure", pixel_stats_exposure,
                        "The value at the top of the color map is 1 / exposure.", 0.1f);
    if (config.config_bool("reset")) {
        pixel_stats_dirty = true;
    }
}

void PathDebugNode::properties_sphere(Properties& config) {
    std::vector<std::string> sphere_options{"off"};
    sphere_options.insert(sphere_options.end(), MAP_CHANNEL_NAMES.begin(), MAP_CHANNEL_NAMES.end());
    int32_t sphere_option = sphere_channel + 1;
    config.config_options("panel", sphere_option, sphere_options, Properties::OptionsStyle::COMBO,
                          "A panel drawn on a sphere at the selected point.");
    sphere_channel = sphere_option - 1;
    if (sphere_channel < 0) {
        return;
    }
    if (!maps_at_point()) {
        config.output_text("point selection required");
    }
    config.config_percent("size", sphere_size, "Of the distance.");
    config.config_bool("scale", sphere_scale,
                       "The radius follows the value, as a polar plot; tests keep the sphere.");
}

void PathDebugNode::properties_reference(Properties& config) {
    if (reference_input_connected) {
        config.output_text("reference: input");
    } else if (reference_loaded) {
        config.output_text("reference: {}", reference_path);
    } else {
        config.output_text("no reference · connect the reference input or set a file");
    }
    if (config.config_path("file", reference_path,
                           "Used while the reference input is not connected.")) {
        reference_dirty = true;
    }
    if (!reference_path.empty() && !reference_loaded) {
        config.output_text("cannot load {}", reference_path);
    }
    if (backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_DIFFERENCE)) {
        config.config_float("gain", difference_gain, "Of the difference image.", 0.1f);
    }
    if (backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_SPLIT_REFERENCE)) {
        config.config_percent("split position", backdrop_split);
    }
    if (shows_reference() && backdrop != static_cast<int32_t>(PATH_DEBUG_BACKDROP_DIFFERENCE)) {
        config.config_bool("alternate", reference_flip,
                           "Blinks between source input and reference.");
        if (reference_flip) {
            config.config_float("rate", reference_flip_hz, "In Hz.", 0.1f);
        }
    }
}

void PathDebugNode::properties_panels(Properties& config) {
    config.config_bool("show", panels_enabled,
                       "While a point is selected; scroll over them to page.");
    for (uint32_t channel = 0; channel < PATH_DEBUG_MAP_CHANNEL_COUNT; channel++) {
        config.config_bool(MAP_CHANNEL_NAMES[channel], panel_enabled[channel],
                           MAP_CHANNEL_DESCRIPTIONS[channel]);
    }
    config.config_int("size", panel_size, "Pixels at a display scale of 1.", 64, 1024);
    config.config_float("exposure", map_exposure, "", 0.01f);
}

void PathDebugNode::properties_bins(Properties& config) {
    maps_dirty |= config.config_options("frame", map_frame, options(FRAME_NAMES),
                                        Properties::OptionsStyle::COMBO);
    maps_dirty |= config.config_options("projection", map_transform, options(TRANSFORM_NAMES),
                                        Properties::OptionsStyle::COMBO);
    if (config.config_int("log2 resolution", map_res_log2, "", MAP_RES_LOG2_MIN,
                          MAP_RES_LOG2_MAX)) {
        maps_dirty = true;
        pending_reconnect = true;
    }
}

void PathDebugNode::properties_sampling(Properties& config) {
    config.config_bool("accumulate", map_accumulate, "Over the frames, else this frame's draws.");
    config.config_int("paths per frame", analysis_paths, "Without a point; 0 = all.", 0, 1 << 22);
    maps_dirty |=
        config.config_int("batch frames", map_batch_frames,
                          "Frames per batch for the variance of reused draws (ReSTIR).", 1, 4096);
    if (config.config_bool("reset")) {
        maps_dirty = true;
    }
}

void PathDebugNode::properties_bsdf_check(Properties& config) {
    maps_dirty |= config.config_options("incident direction", wi_mode, options(WI_NAMES),
                                        Properties::OptionsStyle::COMBO,
                                        "Where the BSDF references are evaluated from.");
    if (wi_mode == static_cast<int32_t>(PATH_DEBUG_WI_MANUAL)) {
        maps_dirty |= config.config_float("polar angle", wi_theta_deg, "Degrees.", 1.f);
        maps_dirty |= config.config_float("azimuth", wi_phi_deg, "Degrees.", 1.f);
        wi_theta_deg = std::clamp(wi_theta_deg, 0.f, 179.f);
        wi_phi_deg = std::clamp(wi_phi_deg, -180.f, 180.f);
    }
    check_requested |= config.config_bool(
        "check", "Integrates the BSDF over the sphere and tops the draws of BSDF sample() up over "
                 "the next frames.");
}

void PathDebugNode::properties_capture(Properties& config) {
    config.config_options("state", capture, options(CAPTURE_STATE_NAMES),
                          Properties::OptionsStyle::COMBO,
                          "Frozen holds the recorded paths; off records, analyzes and draws "
                          "nothing.");
}

void PathDebugNode::properties_export(Properties& config) {
    std::ignore = config.config_text("filename", export_path, false, "Without extension.");
    config.config_options("format", export_format, image_export_format_names(),
                          Properties::OptionsStyle::COMBO);
    config.config_bool("alpha", export_keep_alpha);
    config.config_bool("metadata", export_metadata, "The merian version and the graph config.");
    export_next |= config.config_bool("export");
}

void PathDebugNode::properties_keys(Properties& config) {
    config.config_options("modifier", modifier, options(MODIFIER_NAMES),
                          Properties::OptionsStyle::COMBO,
                          "Held for every binding, so that the scene keeps its own.");
    config.output_text(
        "{0}+click select · {0}+scroll region radius · {0}+Backspace clear selection",
        modifier_name());
}

} // namespace merian
