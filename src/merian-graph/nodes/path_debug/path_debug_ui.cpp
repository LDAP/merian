#include "merian-graph/nodes/path_debug/path_debug.hpp"

#include "merian-shaders/debug/path_record.hpp"
#include "merian/utils/colors.hpp"
#include "merian/utils/properties_imgui.hpp"
#include "merian/vk/imgui/imgui_colormaps.hpp"

#include <fmt/format.h>
#include <imgui.h>

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <numbers>
#include <string>
#include <vector>

namespace merian {

namespace {

using Stats = std::array<uint32_t, PATH_DEBUG_STATS_UINTS>;

float stat_float(const Stats& stats, const uint32_t slot) {
    return std::bit_cast<float>(stats[slot]);
}

float3 stat_rgb(const Stats& stats, const uint32_t slot) {
    return {stat_float(stats, slot), stat_float(stats, slot + 1), stat_float(stats, slot + 2)};
}

float total_variation_percent(const Stats& stats, const uint32_t slot) {
    return std::min(0.5f * stat_float(stats, slot), 1.f) * 100.f;
}

float3 check_albedo(const Stats& stats) {
    return stat_rgb(stats, PATH_DEBUG_CHECK_WEIGHT) /
           static_cast<float>(std::max(stats[PATH_DEBUG_CHECK_DRAWS], 1u));
}

void shadowed_text(ImDrawList* const dl,
                   const ImVec2 pos,
                   const char* text,
                   const ImU32 color = IM_COL32(235, 235, 235, 255)) {
    dl->AddText(ImVec2(pos.x + 1.f, pos.y + 1.f), IM_COL32(0, 0, 0, 200), text);
    dl->AddText(pos, color, text);
}

// chi2/dof of a correct sampler is 1 with a standard deviation of sqrt(2 / dof)
float chi2_sigma(const uint32_t dof) {
    return std::sqrt(2.f / static_cast<float>(std::max(dof, 1u)));
}

ImVec4 chi2_color(const float reduced, const uint32_t dof) {
    const float deviation = std::abs(reduced - 1.f) / chi2_sigma(dof);
    if (deviation < 3.f) {
        return {0.35f, 0.85f, 0.4f, 1.f};
    }
    if (deviation < 5.f) {
        return {0.95f, 0.75f, 0.3f, 1.f};
    }
    return {0.95f, 0.35f, 0.3f, 1.f};
}

std::string si_count(const uint64_t n) {
    if (n >= 1'000'000'000) {
        return fmt::format("{:.1f}G", static_cast<double>(n) / 1e9);
    }
    if (n >= 1'000'000) {
        return fmt::format("{:.1f}M", static_cast<double>(n) / 1e6);
    }
    if (n >= 10'000) {
        return fmt::format("{:.0f}k", static_cast<double>(n) / 1e3);
    }
    return fmt::format("{}", n);
}

void wrapped_text(const std::string& text) {
    ImGui::PushTextWrapPos(0.f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
}

bool direction_disc(const char* id, float& theta_deg, float& phi_deg) {
    constexpr float RADIUS = 76.f;
    constexpr float PAD = 4.f;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 center(origin.x + RADIUS + PAD, origin.y + RADIUS + PAD);
    ImGui::InvisibleButton(id, ImVec2(2.f * (RADIUS + PAD), 2.f * (RADIUS + PAD)));
    const bool below = theta_deg > 90.f;

    bool changed = false;
    if (ImGui::IsItemActive()) {
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        const float dx = (mouse.x - center.x) / RADIUS;
        const float dy = (mouse.y - center.y) / RADIUS;
        const float r = std::sqrt((dx * dx) + (dy * dy));
        if (r > 1e-3f) {
            phi_deg = std::atan2(-dy, dx) * 180.f / std::numbers::pi_v<float>;
            const float polar = std::min(r, 1.f) * 90.f;
            theta_deg = below ? 180.f - polar : polar;
            changed = true;
        }
    }

    ImDrawList* const dl = ImGui::GetWindowDrawList();
    dl->AddCircleFilled(center, RADIUS, IM_COL32(28, 28, 32, 255), 64);
    for (uint32_t ring = 1; ring <= 3; ring++) {
        dl->AddCircle(center, RADIUS * static_cast<float>(ring) / 3.f, IM_COL32(90, 90, 96, 255),
                      64);
    }
    dl->AddLine(ImVec2(center.x - RADIUS, center.y), ImVec2(center.x + RADIUS, center.y),
                IM_COL32(70, 70, 76, 255));
    dl->AddLine(ImVec2(center.x, center.y - RADIUS), ImVec2(center.x, center.y + RADIUS),
                IM_COL32(70, 70, 76, 255));
    shadowed_text(dl, ImVec2(center.x + RADIUS - 24.f, center.y - 16.f), "0°");
    shadowed_text(dl, ImVec2(center.x + 4.f, center.y - RADIUS - 2.f), "90°");

    const float polar = below ? 180.f - theta_deg : theta_deg;
    const float rad = phi_deg * std::numbers::pi_v<float> / 180.f;
    const float handle_r = RADIUS * std::clamp(polar / 90.f, 0.f, 1.f);
    const ImVec2 handle(center.x + (handle_r * std::cos(rad)),
                        center.y - (handle_r * std::sin(rad)));
    dl->AddLine(center, handle, IM_COL32(200, 200, 90, 180), 1.5f);
    if (below) {
        dl->AddCircle(handle, 6.f, IM_COL32(120, 170, 255, 255), 16, 2.f);
    } else {
        dl->AddCircleFilled(handle, 5.f, IM_COL32(240, 220, 90, 255), 16);
    }
    return changed;
}

struct BSDFVerdict {
    enum class Status { ok, fail, no_statement };
    const char* label;
    Status status;
    std::string detail;
};

void check_row(const BSDFVerdict& verdict) {
    ImGui::TextUnformatted(verdict.label);
    ImGui::SameLine(210.f);
    switch (verdict.status) {
    case BSDFVerdict::Status::ok:
        ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.4f, 1.f), "OK (%s)", verdict.detail.c_str());
        break;
    case BSDFVerdict::Status::fail:
        ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.3f, 1.f), "FAIL (%s)", verdict.detail.c_str());
        break;
    default:
        ImGui::TextDisabled("n/a (%s)", verdict.detail.c_str());
        break;
    }
}

std::vector<BSDFVerdict> bsdf_verdicts(const Stats& stats) {
    const uint32_t draws = stats[PATH_DEBUG_CHECK_DRAWS];
    if (draws == 0) {
        return {};
    }

    const float3 albedo_mc = check_albedo(stats);
    const float albedo_lum = yuv_luminance(albedo_mc);
    const float albedo_max = std::max({albedo_mc.x, albedo_mc.y, albedo_mc.z});
    const float albedo_stderr =
        std::sqrt(stat_float(stats, PATH_DEBUG_CHECK_WEIGHT_M2)) / static_cast<float>(draws);
    const uint32_t transmitted = stats[PATH_DEBUG_CHECK_TRANSMITTED];
    const uint32_t bad = stats[PATH_DEBUG_CHECK_BAD];
    const uint32_t bad_cells = stats[PATH_DEBUG_CHECK_BAD_CELLS];
    const bool resolved = stat_float(stats, PATH_DEBUG_CHECK_ALPHA) >= 0.0225f;
    const char* const unresolved = "lobe narrower than the grid";
    const auto identity = [&](const char* label, const uint32_t off_slot, const uint32_t count,
                              const float worst) {
        const float share = static_cast<float>(stats[off_slot]) / static_cast<float>(count);
        return BSDFVerdict{label,
                           share <= 1e-3f ? BSDFVerdict::Status::ok : BSDFVerdict::Status::fail,
                           fmt::format("{:.3f}% off, worst {:.0e}", share * 100.f, worst)};
    };
    const auto integral = [&](const char* label, const bool ok, const std::string& detail) {
        if (!resolved && !ok) {
            return BSDFVerdict{label, BSDFVerdict::Status::no_statement,
                               fmt::format("{}, {}", unresolved, detail)};
        }
        return BSDFVerdict{label, ok ? BSDFVerdict::Status::ok : BSDFVerdict::Status::fail, detail};
    };

    std::vector<BSDFVerdict> verdicts;
    const float mc_tol = std::max(6.f * albedo_stderr, 1e-3f);
    const bool energy_ok = albedo_max <= 1.f + mc_tol;
    verdicts.push_back(
        {"energy conservation", energy_ok ? BSDFVerdict::Status::ok : BSDFVerdict::Status::fail,
         energy_ok ? fmt::format("albedo {:.4f} ± {:.4f}", albedo_max, albedo_stderr)
                   : fmt::format("albedo {:.4f}{}", albedo_max,
                                 transmitted > 0 ? ", refraction scales by eta²" : "")});

    verdicts.push_back(identity("reciprocity", PATH_DEBUG_CHECK_OFF_RECIPROCITY,
                                std::max(stats[PATH_DEBUG_CHECK_RECIPROCITY_CELLS], 1u),
                                stat_float(stats, PATH_DEBUG_CHECK_ERR_RECIPROCITY)));

    const float pdf_integral = stat_float(stats, PATH_DEBUG_CHECK_PDF_INTEGRAL);
    verdicts.push_back(integral("pdf normalized", std::abs(pdf_integral - 1.f) <= 1e-2f,
                                fmt::format("∫pdf dω = {:.4f}", pdf_integral)));

    verdicts.push_back(identity("sample weight = eval / pdf", PATH_DEBUG_CHECK_OFF_WEIGHT, draws,
                                stat_float(stats, PATH_DEBUG_CHECK_ERR_WEIGHT)));
    verdicts.push_back(identity("sampled pdf = pdf()", PATH_DEBUG_CHECK_OFF_PDF, draws,
                                stat_float(stats, PATH_DEBUG_CHECK_ERR_PDF)));

    const bool finite_ok = bad == 0 && bad_cells == 0;
    verdicts.push_back(
        {"finite and non-negative", finite_ok ? BSDFVerdict::Status::ok : BSDFVerdict::Status::fail,
         finite_ok ? fmt::format("{} samples", si_count(draws))
                   : fmt::format("{} samples, {} cells", si_count(bad), si_count(bad_cells))});

    const float eval_lum = yuv_luminance(stat_rgb(stats, PATH_DEBUG_CHECK_EVAL_INTEGRAL));
    const float eval_rel =
        std::abs(eval_lum - albedo_lum) / std::max({eval_lum, albedo_lum, 1e-6f});
    verdicts.push_back(
        integral("sampling integrates eval", eval_rel <= 0.02f,
                 fmt::format("∫f·cos dω {:.4f} vs sampled {:.4f}", eval_lum, albedo_lum)));

    const float albedo_fn = yuv_luminance(stat_rgb(stats, PATH_DEBUG_CHECK_ALBEDO));
    const float albedo_rel =
        std::abs(albedo_fn - albedo_lum) / std::max({albedo_fn, albedo_lum, 1e-6f});
    verdicts.push_back({"get_albedo()",
                        albedo_rel <= 0.15f ? BSDFVerdict::Status::ok : BSDFVerdict::Status::fail,
                        fmt::format("{:.4f} vs {:.4f}", albedo_fn, albedo_lum)});
    return verdicts;
}

void vertex_row(const uint32_t index, const uint32_t* const words) {
    const auto word_f = [&](const uint32_t offset) { return std::bit_cast<float>(words[offset]); };
    const uint32_t meta = words[PATH_RECORD_VERTEX_META];
    const bool terminal = (meta & PATH_RECORD_FLAG_TERMINAL) != 0;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::Text("%u", index);
    ImGui::TableNextColumn();
    const std::string event =
        terminal ? "end"
        : (meta & PATH_RECORD_FLAG_NEE) != 0
            ? "NEE"
            : path_event_name((meta >> PATH_RECORD_LOBE_SHIFT) & PATH_RECORD_LOBE_MASK);
    ImGui::TextUnformatted(event.c_str());
    ImGui::TableNextColumn();
    const float pdf = word_f(PATH_RECORD_VERTEX_PDF);
    const bool resampled = (meta & PATH_RECORD_FLAG_RESAMPLED) != 0;
    if (resampled && !(pdf > 0.f)) {
        ImGui::TextDisabled("-");
    } else if ((meta & PATH_RECORD_FLAG_NEE) != 0 && !resampled && !(pdf > 0.f)) {
        ImGui::TextDisabled("no sample");
    } else if ((meta & (PATH_RECORD_FLAG_REUSED | PATH_RECORD_FLAG_RESAMPLED)) != 0) {
        ImGui::Text("1/W %.4g", pdf);
    } else {
        ImGui::Text("%.4g", pdf);
    }
    ImGui::TableNextColumn();
    ImGui::Text("%.4g", word_f(PATH_RECORD_VERTEX_THROUGHPUT));
    ImGui::TableNextColumn();
    const float3 contrib(word_f(PATH_RECORD_VERTEX_CONTRIB + 0),
                         word_f(PATH_RECORD_VERTEX_CONTRIB + 1),
                         word_f(PATH_RECORD_VERTEX_CONTRIB + 2));
    const float contrib_lum = yuv_luminance(contrib);
    if (std::isfinite(contrib_lum)) {
        ImGui::Text("%.3g", contrib_lum);
    } else {
        ImGui::TextColored(ImVec4(1.f, 0.45f, 0.35f, 1.f), "%.3g, %.3g, %.3g", contrib.x, contrib.y,
                           contrib.z);
    }
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(terminal ? ""
                                    : path_record_method_name((meta >> PATH_RECORD_METHOD_SHIFT) &
                                                              PATH_RECORD_METHOD_MASK));
    ImGui::TableNextColumn();
    const uint32_t material = (meta >> PATH_RECORD_MATERIAL_SHIFT) & PATH_RECORD_MATERIAL_MASK;
    if (!terminal && material != PATH_RECORD_MATERIAL_NONE) {
        ImGui::Text("%u", material);
    }
    ImGui::TableNextColumn();
    const float3 pos(word_f(PATH_RECORD_VERTEX_POS + 0), word_f(PATH_RECORD_VERTEX_POS + 1),
                     word_f(PATH_RECORD_VERTEX_POS + 2));
    if (length(pos) >= PATH_DEBUG_ENV_DISTANCE) {
        ImGui::TextDisabled("environment");
    } else {
        ImGui::Text("%.2f %.2f %.2f", pos.x, pos.y, pos.z);
    }
}

} // namespace

void PathDebugNode::draw_window() {
    ImGui::SetNextWindowPos(ImVec2(440.f, 40.f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(560.f, 0.f), ImVec2(560.f, FLT_MAX));
    if (!ImGui::Begin("Path Debugger", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        const std::scoped_lock lock(stats_mutex);
        mouse_over_ui = ImGui::GetIO().WantCaptureMouse;
        return;
    }

    const std::scoped_lock lock(stats_mutex);
    window_drawn = true;
    mouse_over_ui = ImGui::GetIO().WantCaptureMouse;
    draw_header();

    if (ImGui::BeginTabBar("tabs")) {
        if (ImGui::BeginTabItem("Look")) {
            draw_look_tab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Select")) {
            draw_select_tab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Analyze")) {
            draw_analyze_tab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Capture")) {
            draw_capture_tab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Export")) {
            draw_properties_section(&PathDebugNode::properties_export);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void PathDebugNode::draw_properties_section(void (PathDebugNode::*section)(Properties&)) {
    ImGuiProperties config;
    (this->*section)(config);
}

void PathDebugNode::draw_header() {
    const bool held = freeze && readback_valid && latest_readback.frame.paths > 0;
    if (held) {
        ImGui::TextColored(ImVec4(1.f, 0.76f, 0.25f, 1.f), "FROZEN");
        ImGui::SameLine();
        ImGui::TextDisabled("frame %u", latest_readback.frame.records_frame);
    } else {
        ImGui::TextColored(ImVec4(0.42f, 0.85f, 0.45f, 1.f), "LIVE");
    }
    ImGui::SameLine();
    if (ImGui::SmallButton(freeze ? "resume" : "freeze")) {
        freeze = !freeze;
    }
    if (readback_valid) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s paths · drawing %s", si_count(latest_readback.frame.paths).c_str(),
                            overlay_summary().c_str());
    }
    if (picked) {
        const auto restrict = static_cast<uint32_t>(restrict_mode);
        if (restrict == PATH_DEBUG_RESTRICT_PIXEL) {
            ImGui::Text("pixel (%d, %d)", picked_pixel.x, picked_pixel.y);
        } else if (restrict == PATH_DEBUG_RESTRICT_NONE) {
            ImGui::Text("pixel (%d, %d) · all paths", picked_pixel.x, picked_pixel.y);
        } else {
            ImGui::Text("pixel (%d, %d) · %s", picked_pixel.x, picked_pixel.y,
                        RESTRICT_NAMES[std::min(restrict, PATH_DEBUG_RESTRICT_TO)]);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("clear")) {
            clear_selection();
        }
    }
    ImGui::Separator();
}

std::string PathDebugNode::overlay_summary() const {
    if (isolate_path >= 0) {
        return fmt::format("path #{}", isolate_path);
    }
    if (overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_RESAMPLED)) {
        return fmt::format("{} resampled from {}", latest_readback.frame.draw_count,
                           si_count(resampled_from));
    }
    if (overlay_rank == static_cast<int32_t>(PATH_DEBUG_RANK_RECENT)) {
        return fmt::format("{} most recent", latest_readback.frame.draw_count);
    }
    const uint32_t matched = latest_readback.frame.matched;
    if (matched == 0) {
        return "none";
    }
    const uint32_t cap = params.overlay_max_paths;
    if (matched <= cap) {
        return fmt::format("all {}", si_count(matched));
    }
    switch (static_cast<uint32_t>(overlay_rank)) {
    case PATH_DEBUG_RANK_BRIGHTEST:
        return fmt::format("{} brightest of {}", si_count(cap), si_count(matched));
    case PATH_DEBUG_RANK_FIREFLIES:
        return fmt::format("brightest {:.2g}% of {}", overlay_brightest_fraction * 100.f,
                           si_count(matched));
    default:
        return fmt::format("~{} of {}", si_count(cap), si_count(matched));
    }
}

void PathDebugNode::draw_look_tab() {
    draw_properties_section(&PathDebugNode::properties_overlay);
    if (overlay_color == static_cast<int32_t>(PATH_DEBUG_COLOR_TECHNIQUE)) {
        for (const uint32_t method :
             {PATH_RECORD_METHOD_BSDF, PATH_RECORD_METHOD_GUIDING, PATH_RECORD_METHOD_LIGHT,
              PATH_RECORD_METHOD_RESTIR, PATH_RECORD_METHOD_UNKNOWN}) {
            const uint32_t rgb = PATH_DEBUG_METHOD_COLORS[method];
            ImGui::TextColored(
                ImVec4(static_cast<float>((rgb >> 16) & 0xFFu) / 255.f,
                       static_cast<float>((rgb >> 8) & 0xFFu) / 255.f,
                       static_cast<float>(rgb & 0xFFu) / 255.f, 1.f),
                "%s",
                method == PATH_RECORD_METHOD_UNKNOWN ? "other" : PATH_RECORD_METHOD_NAMES[method]);
            ImGui::SameLine();
        }
        ImGui::NewLine();
    }
    draw_properties_section(&PathDebugNode::properties_backdrop);
}

void PathDebugNode::draw_select_tab() {
    draw_properties_section(&PathDebugNode::properties_selection);
}

void PathDebugNode::draw_capture_tab() {
    if (readback_valid) {
        ImGui::Text("%s of %s paths · keep %.3g", si_count(latest_readback.frame.paths).c_str(),
                    si_count(record_path_capacity).c_str(), params.keep_prob);
    }
    draw_properties_section(&PathDebugNode::properties_capture);
    draw_properties_section(&PathDebugNode::properties_heat);
    if (readback_valid && heat_mode != static_cast<int32_t>(PATH_DEBUG_HEAT_OFF)) {
        const uint32_t cells = latest_readback.frame.grid_cells;
        const uint32_t dropped = latest_readback.frame.grid_dropped;
        ImGui::Text("heat grid %.0f%% full",
                    100.f * static_cast<float>(cells) /
                        static_cast<float>(std::max(params.grid_slots, 1u)));
        if (dropped > 0) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.3f, 1.f), "%s splats dropped",
                               si_count(dropped).c_str());
        }
    }
}

void PathDebugNode::draw_analyze_tab() {
    if (!readback_valid || !ImGui::BeginTabBar("analyze")) {
        return;
    }
    if (ImGui::BeginTabItem("convergence")) {
        draw_convergence_tab();
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("distributions")) {
        draw_distributions_tab();
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("BSDF")) {
        draw_bsdf_tab();
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("paths")) {
        draw_inspector();
        ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("maps")) {
        draw_maps_tab();
        ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
}

void PathDebugNode::draw_convergence_tab() {
    convergence_drawn = true;
    const auto& stats = latest_readback.stats;
    ImGui::SeparatorText(picked ? "picked pixel" : "image centre");
    if (!mean_history.empty()) {
        const std::vector<float> means(mean_history.begin(), mean_history.end());
        const std::vector<float> errs(rel_error_history.begin(), rel_error_history.end());
        const std::vector<float> vars(variance_history.begin(), variance_history.end());
        ImGui::Text("mean %.4g · variance %.4g · rel. error %.3f · %u paths", means.back(),
                    vars.back(), errs.back(), latest_readback.probe.picked_count);
        ImGui::PlotLines("mean", means.data(), static_cast<int>(means.size()), 0, nullptr, FLT_MAX,
                         FLT_MAX, ImVec2(0, 50));
        ImGui::PlotLines("variance", vars.data(), static_cast<int>(vars.size()), 0, nullptr, 0.f,
                         FLT_MAX, ImVec2(0, 50));
        ImGui::PlotLines("rel. error", errs.data(), static_cast<int>(errs.size()), 0, nullptr, 0.f,
                         FLT_MAX, ImVec2(0, 50));
    }

    std::array<float, THETA_BANDS> records_curve{};
    std::array<float, THETA_BANDS> pdf_curve{};
    double records_norm = 0.;
    double pdf_norm = 0.;
    for (uint32_t band = 0; band < THETA_BANDS; band++) {
        const uint32_t base = PATH_DEBUG_STATS_THETA + (3 * band);
        const float omega = stat_float(stats, base + 2);
        if (omega <= 0.f) {
            continue;
        }
        records_curve[band] = static_cast<float>(stats[base + 0]) / omega;
        pdf_curve[band] = stat_float(stats, base + 1) / omega;
        records_norm += static_cast<double>(stats[base + 0]);
        pdf_norm += static_cast<double>(stat_float(stats, base + 1));
    }
    ImGui::SeparatorText("density and pdf over the polar angle");
    ImDrawList* const dl = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float plot_w = ImGui::GetContentRegionAvail().x - 10.f;
    const float plot_h = 110.f;
    dl->AddRectFilled(origin, ImVec2(origin.x + plot_w, origin.y + plot_h),
                      ImGui::GetColorU32(ImGuiCol_FrameBg), 4.f);
    const auto draw_curve = [&](const std::array<float, THETA_BANDS>& curve, const double norm,
                                const ImU32 color, const float thickness) {
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
            const float x = origin.x + ((static_cast<float>(band) + 0.5f) / THETA_BANDS) * plot_w;
            const float y = origin.y + plot_h - 6.f -
                            (static_cast<float>(curve[band] / norm) / vmax) * (plot_h - 12.f);
            points[band] = ImVec2(x, y);
        }
        dl->AddPolyline(points.data(), THETA_BANDS, color, ImDrawFlags_None, thickness);
    };
    draw_curve(pdf_curve, pdf_norm, IM_COL32(235, 235, 235, 255), 4.f);
    draw_curve(records_curve, records_norm, IM_COL32(203, 166, 247, 255), 1.5f);
    ImGui::Dummy(ImVec2(plot_w, plot_h + 4.f));
    ImGui::TextColored(ImVec4(0.92f, 0.92f, 0.92f, 1.f), "pdf");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.80f, 0.65f, 0.97f, 1.f), "records");
}

void PathDebugNode::draw_distributions_tab() const {
    const auto& stats = latest_readback.stats;
    const auto histogram = [&](const char* label, const uint32_t offset, const uint32_t bins) {
        std::array<float, PATH_DEBUG_HIST_SMALL_BINS> values{};
        float vmax = 0.f;
        for (uint32_t i = 0; i < bins; i++) {
            values[i] = static_cast<float>(stats[offset + i]);
            vmax = std::max(vmax, values[i]);
        }
        ImGui::PlotHistogram(label, values.data(), static_cast<int>(bins), 0, nullptr, 0.f, vmax,
                             ImVec2(0, 60));
    };
    histogram("path luminance (log2)", PATH_DEBUG_STATS_LUM, PATH_DEBUG_HIST_SMALL_BINS);
    histogram("sample pdf (log2)", PATH_DEBUG_STATS_PDF, PATH_DEBUG_HIST_SMALL_BINS);
    histogram("paths by scatter events", PATH_DEBUG_STATS_BOUNCE, PATH_DEBUG_LENGTH_BINS);

    std::array<float, PATH_DEBUG_LENGTH_BINS> contrib{};
    std::array<float, PATH_DEBUG_LENGTH_BINS> mean{};
    float total = 0.f;
    for (uint32_t i = 0; i < PATH_DEBUG_LENGTH_BINS; i++) {
        contrib[i] = stat_float(stats, PATH_DEBUG_STATS_LEN_CONTRIB + i);
        const uint32_t n = stats[PATH_DEBUG_STATS_BOUNCE + i];
        mean[i] = n > 0 ? contrib[i] / static_cast<float>(n) : 0.f;
        total += contrib[i];
    }
    ImGui::PlotHistogram("contribution by scatter events", contrib.data(),
                         static_cast<int>(contrib.size()), 0, nullptr, 0.f, FLT_MAX, ImVec2(0, 60));
    ImGui::PlotHistogram("mean contribution per path", mean.data(), static_cast<int>(mean.size()),
                         0, nullptr, 0.f, FLT_MAX, ImVec2(0, 60));
    if (total > 0.f) {
        std::string shares;
        for (uint32_t i = 0; i < PATH_DEBUG_LENGTH_BINS; i++) {
            if (contrib[i] > 0.02f * total) {
                shares += fmt::format("{}: {:.0f}%  ", i, 100.f * contrib[i] / total);
            }
        }
        wrapped_text(fmt::format("energy share: {}", shares));
    }
}

void PathDebugNode::draw_bsdf_tab() {
    const auto& stats = latest_readback.stats;
    if (!gbuffer_connected) {
        ImGui::TextDisabled("needs the gbuffer input");
        return;
    }
    if (!picked) {
        ImGui::TextDisabled("no pixel picked");
        return;
    }

    // 1. the incident direction every tool below uses
    ImGui::SeparatorText("incident direction");
    bool wi_changed = false;
    for (int32_t mode = 0; mode < static_cast<int32_t>(WI_NAMES.size()); mode++) {
        if (mode > 0) {
            ImGui::SameLine();
        }
        if (ImGui::RadioButton(WI_NAMES[mode], wi_mode == mode)) {
            wi_mode = mode;
            wi_changed = true;
        }
    }
    if (direction_disc("wi", wi_theta_deg, wi_phi_deg)) {
        wi_mode = static_cast<int32_t>(PATH_DEBUG_WI_MANUAL);
        wi_changed = true;
    }
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::PushItemWidth(150.f);
    wi_changed |= ImGui::DragFloat("polar angle", &wi_theta_deg, 0.5f, 0.f, 179.f, "%.1f°");
    wi_changed |= ImGui::DragFloat("azimuth", &wi_phi_deg, 0.5f, -180.f, 180.f, "%.1f°");
    ImGui::PopItemWidth();
    bool below = wi_theta_deg > 90.f;
    if (ImGui::Checkbox("below the surface", &below)) {
        wi_theta_deg = 180.f - wi_theta_deg;
        wi_changed = true;
    }
    ImGui::EndGroup();
    if (wi_changed) {
        maps_dirty = true;
    }

    // 2. the validation suite
    ImGui::SeparatorText("validation");
    if (ImGui::Button("check BSDF")) {
        check_requested = true;
    }
    const std::vector<BSDFVerdict> verdicts = bsdf_verdicts(stats);
    if (verdicts.empty()) {
        return;
    }
    for (const BSDFVerdict& verdict : verdicts) {
        check_row(verdict);
    }

    const float draws = static_cast<float>(stats[PATH_DEBUG_CHECK_DRAWS]);
    const float3 albedo_mc = check_albedo(stats);
    ImGui::Text("alpha %.3g · %.1f%% valid · %.1f%% transmitted",
                stat_float(stats, PATH_DEBUG_CHECK_ALPHA),
                100.f * static_cast<float>(stats[PATH_DEBUG_CHECK_VALID]) / draws,
                100.f * static_cast<float>(stats[PATH_DEBUG_CHECK_TRANSMITTED]) / draws);
    ImGui::Text("albedo (%.4f, %.4f, %.4f)", albedo_mc.x, albedo_mc.y, albedo_mc.z);

    // 3. total variation against the pdf
    ImGui::SeparatorText("total variation vs pdf");
    if (stats[PATH_DEBUG_STATS_CHECK_SAMPLES] > 0) {
        ImGui::Text("sampled %.2f%%", total_variation_percent(stats, PATH_DEBUG_STATS_TV_CHECK));
        ImGui::SameLine();
    }
    ImGui::Text("eval %.2f%%", total_variation_percent(stats, PATH_DEBUG_STATS_TV_EVAL));
    if (shared_support(true)) {
        ImGui::SameLine();
        ImGui::Text("records %.2f%%", total_variation_percent(stats, PATH_DEBUG_STATS_TV_RECORDS));
    }
}

void PathDebugNode::draw_inspector() {
    inspector_drawn = true;

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
            if (entry.index == PATH_DEBUG_NO_ISOLATE) {
                break;
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool selected = isolate_path == static_cast<int32_t>(entry.index);
            if (ImGui::Selectable(fmt::format("#{}##path{}", entry.index, t).c_str(), selected,
                                  ImGuiSelectableFlags_SpanAllColumns)) {
                isolate_path = selected ? -1 : static_cast<int32_t>(entry.index);
            }
            ImGui::TableNextColumn();
            ImGui::Text("%u, %u", entry.pixel & 0xFFFFu, entry.pixel >> 16);
            ImGui::TableNextColumn();
            ImGui::Text("%.4g", std::bit_cast<float>(entry.luminance));
            ImGui::TableNextColumn();
            const std::string expression =
                path_expression_of(entry.classes, entry.sample_and_scatter >> 16);
            ImGui::TextUnformatted(expression.c_str());
            ImGui::SameLine();
            if (ImGui::SmallButton(fmt::format("filter##expr{}", t).c_str())) {
                set_expression(false, expression);
                scatter_range = int2(0, 32);
                method_mask = 0;
                material_id = -1;
            }
        }
        ImGui::EndTable();
    }

    if (isolate_path < 0) {
        return;
    }
    if (ImGui::Button("show all")) {
        isolate_path = -1;
        return;
    }

    const IsolatedPath& isolated = latest_readback.isolated;
    if (isolated.valid == 0) {
        ImGui::TextDisabled("#%d not recorded", isolate_path);
        return;
    }
    const uint32_t vertices = isolated.sample_and_count >> 16;
    ImGui::Text("pixel (%u, %u) · sample %u · frame %u · luminance %.4g", isolated.pixel & 0xFFFFu,
                isolated.pixel >> 16, isolated.sample_and_count & 0xFFFFu,
                latest_readback.frame.records_frame, std::bit_cast<float>(isolated.luminance));

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
            vertex_row(v, isolated.vertices.data() + (std::size_t{v} * PATH_RECORD_VERTEX_UINTS));
        }
        ImGui::EndTable();
    }
}

void PathDebugNode::draw_maps_tab() {
    draw_sampling_fit();
    draw_properties_section(&PathDebugNode::properties_maps);
    draw_properties_section(&PathDebugNode::properties_pixel_stats);
}

void PathDebugNode::draw_sampling_fit() const {
    const auto& stats = latest_readback.stats;
    ImGui::Text("%s samples · pdf integral %.3f", si_count(stats[PATH_DEBUG_STATS_SAMPLES]).c_str(),
                stat_float(stats, PATH_DEBUG_STATS_PDF_INTEGRAL));
    const bool recorded = map_pdf_ref == static_cast<int32_t>(PATH_DEBUG_PDF_REF_RECORDED);
    const uint32_t dof = stats[PATH_DEBUG_STATS_CHI2_DOF];
    const uint32_t seen = stats[PATH_DEBUG_STATS_SEEN];
    const uint32_t batches = stats[PATH_DEBUG_STATS_BATCHES];
    if (!shared_support(!recorded)) {
        ImGui::TextDisabled(shared_support(false)
                                ? "chi2 needs the camera ray as incident direction"
                                : "chi2 needs the picked pixel at bounce 0");
    } else if ((seen & PATH_DEBUG_SEEN_RESAMPLED) != 0) {
        ImGui::TextDisabled("chi2: resampled paths have no per-vertex density");
    } else if (!recorded && (seen & PATH_DEBUG_SEEN_CONNECTION) != 0) {
        ImGui::TextDisabled("chi2 of light draws needs the recorded pdf reference");
    } else if ((seen & PATH_DEBUG_SEEN_REUSED) != 0 && batches < PATH_DEBUG_MIN_BATCHES) {
        ImGui::TextDisabled("chi2 of reused samples: batch %u of %u", batches,
                            PATH_DEBUG_MIN_BATCHES);
    } else if (dof > 0) {
        const float reduced = stat_float(stats, PATH_DEBUG_STATS_CHI2) / static_cast<float>(dof);
        ImGui::TextColored(chi2_color(reduced, dof),
                           "chi2/dof %.2f over %u bins, expected 1 ± %.2f", reduced, dof,
                           chi2_sigma(dof));
    } else {
        ImGui::TextDisabled("chi2: too few samples");
    }
    if ((seen & PATH_DEBUG_SEEN_REUSED) != 0 && batches > 1) {
        const float lag_square = stat_float(stats, PATH_DEBUG_STATS_LAG_SQUARE);
        const float correlation =
            lag_square > 0.f ? stat_float(stats, PATH_DEBUG_STATS_LAG_PRODUCT) / lag_square : 0.f;
        const std::string batch_text =
            fmt::format("{} batches of {} frames · lag-1 correlation {:.2f}", batches,
                        map_batch_frames, correlation);
        if (correlation > 0.2f) {
            ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.3f, 1.f), "%s: raise batch frames",
                               batch_text.c_str());
        } else {
            ImGui::TextDisabled("%s", batch_text.c_str());
        }
    }
    if (!recorded && stats[PATH_DEBUG_STATS_SPIKE_BINS] > 0) {
        ImGui::TextDisabled("%.1f%% of samples in %u quasi-delta bins, excluded",
                            stat_float(stats, PATH_DEBUG_STATS_SPIKE_MASS) * 100.f,
                            stats[PATH_DEBUG_STATS_SPIKE_BINS]);
    }
    ImGui::Separator();
}

void PathDebugNode::draw_overlay() {
    const std::scoped_lock lock(stats_mutex);
    draw_panel_labels();
    draw_probe_tooltip();
}

void PathDebugNode::draw_panel_labels() const {
    if (!panels_enabled) {
        return;
    }
    ImDrawList* const dl = ImGui::GetBackgroundDrawList();
    const auto& stats = latest_readback.stats;
    const float panel = static_cast<float>(params.map_panel_size);
    const float x0 = static_cast<float>(panel_left());
    const float shared_max = std::max({stat_float(stats, PATH_DEBUG_STATS_MAX_DENSITY),
                                       stat_float(stats, PATH_DEBUG_STATS_MAX_PDF),
                                       stat_float(stats, PATH_DEBUG_STATS_MAX_CHECK),
                                       stat_float(stats, PATH_DEBUG_STATS_MAX_EVAL)});
    const uint32_t dof = stats[PATH_DEBUG_STATS_CHI2_DOF];

    uint32_t slot = 0;
    for (uint32_t channel = 0; channel < MAP_CHANNEL_COUNT; channel++) {
        if ((params.map_panel_mask & (1u << channel)) == 0) {
            continue;
        }
        const float y0 = static_cast<float>(panel_top(slot++));
        dl->AddRect(ImVec2(x0 - 1.f, y0 - 1.f), ImVec2(x0 + panel + 1.f, y0 + panel + 1.f),
                    IM_COL32(90, 90, 90, 255));
        if (channel != PATH_DEBUG_MAP_CONTRIB_COLOR) {
            const bool diverging =
                channel == PATH_DEBUG_MAP_RATIO || channel == PATH_DEBUG_MAP_ZSCORE;
            imgui_colorbar(dl, ImVec2(x0, y0 + panel + 1.f), ImVec2(x0 + panel, y0 + panel + 4.f),
                           diverging ? imgui_colormap_diverging : imgui_colormap_turbo);
        }

        std::string value;
        ImU32 value_color = IM_COL32(235, 235, 235, 255);
        switch (channel) {
        case PATH_DEBUG_MAP_DENSITY:
        case PATH_DEBUG_MAP_PDF:
            value = fmt::format("max {:.2g}/sr", shared_max);
            break;
        case PATH_DEBUG_MAP_CONTRIB:
            value = fmt::format("max {:.2g}/sr", stat_float(stats, PATH_DEBUG_STATS_MAX_CONTRIB));
            break;
        case PATH_DEBUG_MAP_MEAN:
            value = fmt::format("max {:.2g}", stat_float(stats, PATH_DEBUG_STATS_MAX_MEAN));
            break;
        case PATH_DEBUG_MAP_RATIO:
            value = "0.25x - 4x";
            break;
        case PATH_DEBUG_MAP_ZSCORE:
            if (dof > 0) {
                const float reduced =
                    stat_float(stats, PATH_DEBUG_STATS_CHI2) / static_cast<float>(dof);
                value = fmt::format("chi2/dof {:.2f}", reduced);
                value_color = ImGui::ColorConvertFloat4ToU32(chi2_color(reduced, dof));
            }
            break;
        case PATH_DEBUG_MAP_CHECK:
            if (stats[PATH_DEBUG_STATS_CHECK_SAMPLES] > 0) {
                value = si_count(stats[PATH_DEBUG_STATS_CHECK_SAMPLES]);
            }
            break;
        case PATH_DEBUG_MAP_EVAL:
            value = fmt::format("albedo {:.3g}", stat_float(stats, PATH_DEBUG_STATS_EVAL_NORM));
            break;
        default:
            break;
        }
        const float text_y = y0 + panel + 6.f;
        shadowed_text(dl, ImVec2(x0, text_y), MAP_CHANNEL_NAMES[channel]);
        shadowed_text(dl, ImVec2(x0 + panel - ImGui::CalcTextSize(value.c_str()).x, text_y),
                      value.c_str(), value_color);
    }

    if (panel_overflow > 0) {
        const float height = static_cast<float>(extent.height);
        const float column = height + static_cast<float>(panel_overflow);
        const float track_x = x0 + panel + 4.f;
        const float top = height * static_cast<float>(panel_scroll) / column;
        dl->AddRectFilled(ImVec2(track_x, 0.f), ImVec2(track_x + 3.f, height),
                          IM_COL32(0, 0, 0, 90));
        dl->AddRectFilled(ImVec2(track_x, top),
                          ImVec2(track_x + 3.f, top + (height * height / column)),
                          IM_COL32(200, 200, 200, 200));
    }
}

void PathDebugNode::draw_probe_tooltip() const {
    const Probe& probe = latest_readback.probe;
    if (params.cursor_x == PATH_DEBUG_NO_PIXEL) {
        return;
    }
    std::string text;
    if (probe.kind == PATH_DEBUG_PROBE_PANEL && params.probe_channel != PATH_DEBUG_MAP_NONE) {
        text = fmt::format("density {:.3g}/sr · pdf {:.3g}\nmean {:.3g} · {} samples",
                           std::bit_cast<float>(probe.a), std::bit_cast<float>(probe.b),
                           std::bit_cast<float>(probe.c), probe.count);
    } else if (probe.kind == PATH_DEBUG_PROBE_PIXEL) {
        text = fmt::format("mean {:.4g} · rel. error {:.3f}\nvariance {:.3g} · {} paths",
                           std::bit_cast<float>(probe.a), std::bit_cast<float>(probe.b),
                           std::bit_cast<float>(probe.c), probe.count);
    }
    if (text.empty()) {
        return;
    }
    ImDrawList* const dl = ImGui::GetBackgroundDrawList();
    const ImVec2 size = ImGui::CalcTextSize(text.c_str());
    const float tx = std::min(static_cast<float>(params.cursor_x) + 16.f,
                              static_cast<float>(extent.width) - size.x - 8.f);
    const float ty = std::min(static_cast<float>(params.cursor_y) + 16.f,
                              static_cast<float>(extent.height) - size.y - 6.f);
    dl->AddRectFilled(ImVec2(tx - 5.f, ty - 3.f), ImVec2(tx + size.x + 5.f, ty + size.y + 3.f),
                      IM_COL32(15, 15, 18, 215), 4.f);
    shadowed_text(dl, ImVec2(tx, ty), text.c_str());
}

} // namespace merian
