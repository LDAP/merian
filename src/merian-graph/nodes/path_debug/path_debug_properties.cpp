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
constexpr std::array<const char*, 2> METHOD_MODE_NAMES = {"any vertex", "every vertex"};
constexpr std::array<const char*, 4> RANK_NAMES = {"most recent", "brightest", "brightest share",
                                                   "resampled by contribution"};
constexpr std::array<const char*, 3> DEPTH_NAMES = {"x-ray", "hide", "dash"};
constexpr std::array<const char*, 3> TRANSFORM_NAMES = {"octahedral", "lat-long",
                                                        "hemisphere (equal-area)"};
constexpr std::array<const char*, 2> FRAME_NAMES = {"world (y up)", "local shading"};
constexpr std::array<const char*, 2> PDF_REF_NAMES = {"BSDF pdf()", "recorded pdf"};
constexpr std::array<const char*, 3> FALSE_COLOR_NAMES = {"off", "scene heatmap",
                                                          "pixel statistic"};
// without the off state at 0
constexpr std::array<const char*, 3> HEAT_NAMES = {"vertex density", "downstream contribution",
                                                   "throughput"};
constexpr std::array<const char*, 3> PIXEL_STAT_NAMES = {"mean", "sample variance",
                                                         "relative standard error"};

constexpr Properties::ChildFlags HEADERLESS_OPEN =
    Properties::ChildFlagBits::HEADERLESS | Properties::ChildFlagBits::DEFAULT_OPEN;

constexpr std::array<const char*, PATH_DEBUG_MAP_CHANNEL_COUNT> MAP_CHANNEL_DESCRIPTIONS = {
    "f * L * cos over every technique as a density: what a zero-variance sampler samples.",
    "Directions per steradian of the binned draws.",
    "The pdf the renderer recorded for the binned draws.",
    "The material's pdf for the incident direction.",
    "The material's own sample(), without the renderer.",
    "f * cos, normalized: the distribution an ideal BSDF sampler follows.",
    "Poisson deviation of each bin from the pdf it is tested against; chi2/dof sums them.",
    "Sampled density over the pdf it is tested against: white at 1, blue undersampled, red "
    "oversampled.",
    "Sampled density over the normalized integrand: white at 1, blue undersampled, red "
    "oversampled.",
    "f * L * cos over every technique, per steradian.",
    "The integrand in its own color, on the integrand's brightness scale.",
    "Squared contributions per steradian: where the variance comes from.",
    "The integrand over BSDF eval(): the light arriving at the point.",
    "Mean contribution of a binned draw in each direction.",
    "Mean color of a binned draw's contribution in each direction."};

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
        section("filter", "filter", OPEN, [&] { properties_filter(config); });
        properties_line_style(config);
    });
    section("view", "view", OPEN, [&] { properties_view(config); });
    section("analysis", "analysis", OPEN, [&] {
        section("sampling", "sampling", {}, [&] { properties_sampling(config); });
        section("bsdf check", "BSDF check", {}, [&] { properties_bsdf_check(config); });
    });
    section("capture", "capture", OPEN, [&] { properties_capture(config); });
    section("export", "export", OPEN, [&] { properties_export(config); });
    section("keys", "keys", OPEN, [&] { properties_keys(config); });
    return std::exchange(pending_reconnect, false) ? NodeStatusFlags{NEEDS_RECONNECT}
                                                   : NodeStatusFlags{};
}

void PathDebugNode::properties_selection(Properties& config) {
    config.output_text(selection_status());
    int32_t kind = select_kind;
    if (config.config_options("kind", kind, options(SELECT_NAMES),
                              Properties::OptionsStyle::RADIO_BUTTON,
                              "Point: the pixel's paths that start at the surface. Region: the "
                              "paths through a sphere around it.")) {
        set_select_kind(kind);
    }
    if (select_kind == SELECT_REGION) {
        int32_t match = region_match;
        if (config.config_options("match", match, options(MATCH_NAMES),
                                  Properties::OptionsStyle::RADIO_BUTTON,
                                  "Whether a segment crosses the sphere, or the first or the last "
                                  "vertex lies inside it.")) {
            set_region_match(match);
        }
    }
    if (has_selection &&
        config.config_bool("clear", fmt::format("{}+Esc does the same.", modifier_name()))) {
        clear_selection();
    }
}

void PathDebugNode::properties_selection_details(Properties& config) {
    bool changed = false;
    if (select_kind == SELECT_REGION && selected_pos_valid) {
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
                          "Path luminance: what the path adds to its pixel. Segment contribution: "
                          "the radiance each segment carries to the camera.");
}

void PathDebugNode::properties_filter(Properties& config) {
    std::string text = expression;
    if (config.config_text(
            "expression", text, false,
            "S G D U specular, glossy, diffuse, unclassified; R T reflected, transmitted; . any; "
            "<RD> both; [SG] either; * + ? {n} {n,m}. A trailing N ends at an NEE connection (.N "
            "direct, .*S.+N after a specular event). Empty matches every path.")) {
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
    changed |= config.config_int("material", material_id, "Material id; -1 = any.", -1,
                                 static_cast<int32_t>(PATH_RECORD_MATERIAL_NONE) - 1);
    if (config.st_begin_child("technique", "technique")) {
        for (uint32_t m = 0; m < PATH_RECORD_METHOD_NAMES.size(); m++) {
            bool set = (method_mask & (1u << m)) != 0;
            if (config.config_bool(PATH_RECORD_METHOD_NAMES[m], set)) {
                method_mask = set ? (method_mask | (1u << m)) : (method_mask & ~(1u << m));
                changed = true;
            }
        }
        changed |= config.config_options("match", method_mode, options(METHOD_MODE_NAMES),
                                         Properties::OptionsStyle::COMBO,
                                         "None ticked matches any technique.");
        config.st_end_child();
    }
    if (changed) {
        invalidate_selection();
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
    kept_dirty |= config.config_int("max paths", overlay_max_paths, "", 1,
                                    static_cast<int32_t>(PATH_DEBUG_KEPT_SLOTS));
    overlay_max_paths =
        std::clamp(overlay_max_paths, 1, static_cast<int32_t>(PATH_DEBUG_KEPT_SLOTS));
    if (overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_BRIGHTEST_SHARE)) {
        config.config_float("share", overlay_brightest_share, "Of the frame's paths.", 1e-5f);
        overlay_brightest_share = std::clamp(overlay_brightest_share, 1e-7f, 1.f);
    }
    if (overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_RESAMPLED)) {
        config.config_float("history", resample_history,
                            "Frames over which drawn paths lose weight; 0 = since the last "
                            "change.",
                            1.f);
        resample_history = std::max(resample_history, 0.f);
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

void PathDebugNode::properties_view(Properties& config) {
    config.config_options("image", backdrop, options(BACKDROP_NAMES),
                          Properties::OptionsStyle::COMBO, "What the paths are drawn on.");
    if (config.is_ui() && backdrop_missing) {
        config.output_text(shows_reference() ? "reference required · showing render"
                                             : "backdrop input required · showing render");
    }
    if (backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_FILTERED)) {
        config.config_float("exposure", filtered_exposure,
                            "0 = raw radiance. Accumulates until the selection changes.", 0.01f);
    }
    if (backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_DIFFERENCE)) {
        config.config_float("gain", difference_gain, "", 0.1f);
    }
    if (backdrop == static_cast<int32_t>(PATH_DEBUG_BACKDROP_SPLIT_REFERENCE)) {
        config.config_percent("split position", backdrop_split);
    }
    if (shows_reference() && backdrop != static_cast<int32_t>(PATH_DEBUG_BACKDROP_DIFFERENCE)) {
        config.config_bool("alternate", reference_flip, "Blinks between render and reference.");
        if (reference_flip) {
            config.config_float("rate", reference_flip_hz, "In Hz.", 0.1f);
        }
    }

    config.st_separate("reference");
    if (reference_input_connected) {
        config.output_text("reference: input");
    } else if (reference_loaded) {
        config.output_text("reference: {}", reference_path);
    } else {
        config.output_text("no reference · connect the reference input or set a file");
    }
    if (config.config_path("reference file", reference_path,
                           "Used while the reference input is not connected.")) {
        reference_dirty = true;
    }
    if (!reference_path.empty() && !reference_loaded) {
        config.output_text("cannot load {}", reference_path);
    }

    config.st_separate("false color");
    config.config_options("false color", false_color, options(FALSE_COLOR_NAMES),
                          Properties::OptionsStyle::COMBO,
                          "One quantity tinted over the image; its settings follow.");
    if (false_color != FALSE_COLOR_OFF) {
        config.config_percent("opacity", false_color_opacity);
    }
    if (false_color == FALSE_COLOR_HEAT &&
        config.st_begin_child("heatmap", "heatmap", HEADERLESS_OPEN)) {
        if (!gbuffer_connected) {
            config.output_text("gbuffer required");
        }
        heat_dirty |= config.config_options(
            "quantity", heat_quantity, options(HEAT_NAMES), Properties::OptionsStyle::COMBO,
            "Per world-space cell: the path vertices, the contribution downstream of them, or "
            "their throughput.");
        heat_dirty |= config.config_vec("bounces", heat_bounce_range);
        heat_bounce_range.x = std::clamp(heat_bounce_range.x, 0, HEAT_BOUNCE_MAX);
        heat_bounce_range.y = std::clamp(heat_bounce_range.y, heat_bounce_range.x, HEAT_BOUNCE_MAX);
        heat_dirty |= config.config_float("cell size", heat_cell_size, "World units.", 0.001f);
        heat_cell_size = std::max(heat_cell_size, 1e-4f);
        config.config_float("history", heat_history, "Frames; 0 = keep forever.", 1.f);
        heat_history = std::max(heat_history, 0.f);
        config.config_float("exposure", heat_exposure, "", 0.01f);
        config.config_bool("smooth", heat_smooth);
        if (config.config_int("log2 grid slots", grid_slots_log2, "", GRID_SLOTS_LOG2_MIN,
                              GRID_SLOTS_LOG2_MAX)) {
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
        config.st_end_child();
    }
    if (false_color == FALSE_COLOR_PIXEL_STAT &&
        config.st_begin_child("pixel statistic", "pixel statistic", HEADERLESS_OPEN)) {
        config.config_options("statistic", pixel_statistic, options(PIXEL_STAT_NAMES),
                              Properties::OptionsStyle::COMBO,
                              "Of each pixel's path luminance over frames: mean, variance, and the "
                              "standard error relative to the mean. No reference needed.");
        config.config_float("exposure", pixel_stats_exposure,
                            "The value at the top of the color map is 1 / exposure.", 0.1f);
        if (config.config_bool("reset")) {
            pixel_stats_dirty = true;
        }
        config.st_end_child();
    }

    config.st_separate("sphere");
    std::vector<std::string> sphere_options{"off"};
    sphere_options.insert(sphere_options.end(), MAP_CHANNEL_NAMES.begin(), MAP_CHANNEL_NAMES.end());
    int32_t sphere_option = sphere_channel + 1;
    config.config_options("sphere", sphere_option, sphere_options, Properties::OptionsStyle::COMBO,
                          "A directional map drawn on a sphere at the selected point.");
    sphere_channel = sphere_option - 1;
    if (sphere_channel >= 0 && !maps_at_point()) {
        config.output_text("point selection required");
    }
    if (sphere_channel >= 0) {
        config.config_percent("sphere size", sphere_size, "Of the distance.");
    }
}

void PathDebugNode::properties_sampling(Properties& config) {
    if (!gbuffer_connected) {
        config.output_text("gbuffer required");
    }
    config.st_separate("binning");
    maps_dirty |= config.config_options("binned", map_draws, options(DRAWS_NAMES),
                                        Properties::OptionsStyle::COMBO,
                                        "Scatter: every continuation, recorded with the mixture "
                                        "density. NEE: the light connections.");
    maps_dirty |= config.config_options("frame", map_frame, options(FRAME_NAMES),
                                        Properties::OptionsStyle::COMBO);
    maps_dirty |= config.config_options("projection", map_transform, options(TRANSFORM_NAMES),
                                        Properties::OptionsStyle::COMBO);
    if (config.config_int("log2 resolution", map_res_log2, "", MAP_RES_LOG2_MIN,
                          MAP_RES_LOG2_MAX)) {
        maps_dirty = true;
        pending_reconnect = true;
    }

    config.st_separate("test");
    maps_dirty |= config.config_options("test against", map_pdf_ref, options(PDF_REF_NAMES),
                                        Properties::OptionsStyle::COMBO,
                                        "BSDF pdf(): the material's pdf. Recorded pdf: the pdf "
                                        "each sampler recorded with its sample.");
    maps_dirty |=
        config.config_int("batch frames", map_batch_frames,
                          "Frames per batch for the variance of reused samples.", 1, 4096);

    config.st_separate("accumulation");
    config.config_bool("accumulate", map_accumulate);
    config.config_int("paths per frame", analysis_paths, "0 = all", 0, 1 << 22);
    if (config.config_bool("reset")) {
        maps_dirty = true;
    }

    config.st_separate("panels");
    config.config_bool("show", panels_enabled,
                       "While a point is selected; scroll over them to page.");
    for (uint32_t channel = 0; channel < PATH_DEBUG_MAP_CHANNEL_COUNT; channel++) {
        config.config_bool(MAP_CHANNEL_NAMES[channel], panel_enabled[channel],
                           MAP_CHANNEL_DESCRIPTIONS[channel]);
    }
    config.config_int("size", panel_size, "Pixels at a display scale of 1.", 64, 1024);
    config.config_float("exposure", map_exposure, "", 0.01f);
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
        "check", "Tops the draws of BSDF sample() up and integrates the BSDF over the sphere.");
}

void PathDebugNode::properties_capture(Properties& config) {
    config.config_options("state", capture, options(CAPTURE_STATE_NAMES),
                          Properties::OptionsStyle::COMBO,
                          "Frozen holds the recorded paths; off records, analyzes and draws "
                          "nothing.");
    config.config_bool("auto record probability", auto_keep_prob,
                       "Fits the record probability to the record buffer.");
    if (!auto_keep_prob) {
        config.config_percent("record probability", keep_prob);
        keep_prob = std::clamp(keep_prob, MIN_KEEP_PROB, 1.f);
    }
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
    config.output_text("{0}+click select · {0}+scroll region radius · {0}+Esc clear selection",
                       modifier_name());
}

} // namespace merian
